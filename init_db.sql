-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS ai CASCADE;

-- Core table: prescriptions metadata
CREATE TABLE IF NOT EXISTS prescriptions (
    id              SERIAL PRIMARY KEY,
    file_name       TEXT NOT NULL,
    storage_url     TEXT NOT NULL,
    file_type       TEXT NOT NULL,           -- 'pdf' or 'image'
    raw_text        TEXT,
    summary         TEXT,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Parsed medical entities from each prescription
CREATE TABLE IF NOT EXISTS medical_entities (
    id                  SERIAL PRIMARY KEY,
    prescription_id     INTEGER NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    patient_name        TEXT,
    patient_age         TEXT,
    patient_gender      TEXT,
    doctor_name         TEXT,
    doctor_speciality   TEXT,
    hospital_clinic     TEXT,
    diagnosis           TEXT[],
    symptoms            TEXT[],
    parsed_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Individual medications
CREATE TABLE IF NOT EXISTS medications (
    id                  SERIAL PRIMARY KEY,
    entity_id           INTEGER NOT NULL REFERENCES medical_entities(id) ON DELETE CASCADE,
    drug_name           TEXT NOT NULL,
    dosage              TEXT,
    frequency           TEXT,
    duration            TEXT,
    route               TEXT,
    instructions        TEXT
);

-- Embeddings for semantic search
CREATE TABLE IF NOT EXISTS prescription_embeddings (
    id                  SERIAL PRIMARY KEY,
    prescription_id     INTEGER NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    chunk_text          TEXT NOT NULL,
    embedding           vector(2048)
);

-- Indexes for fast search
CREATE INDEX IF NOT EXISTS idx_prescriptions_file_name ON prescriptions USING gin (file_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_medical_entities_patient ON medical_entities USING gin (patient_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_medical_entities_doctor ON medical_entities USING gin (doctor_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_medical_entities_diagnosis ON medical_entities USING gin (diagnosis);
CREATE INDEX IF NOT EXISTS idx_medications_drug ON medications USING gin (drug_name gin_trgm_ops);
-- Note: pgvector indexes (hnsw/ivfflat) cap at 2000 dims.
-- The auto-vectorizer creates its own indexed embedding table.
-- For manual prescription_embeddings, cosine distance queries use sequential scan.

-- Inject OpenRouter config from container env vars (set by docker-compose from .env)
DO $$
import os
for key in ('openrouter_api_key', 'openrouter_embed_model', 'openrouter_complete_model'):
    val = os.environ.get(key.upper(), '')
    if val:
        plpy.execute(f"ALTER DATABASE postgres SET ai.{key} = '{val}'")
$$ LANGUAGE plpython3u;

-- Custom OpenRouter embedding function (OpenRouter response format not compatible with ai.openai_embed)
CREATE OR REPLACE FUNCTION public.openrouter_embed(
    model_name text,
    input_text text,
    api_key text DEFAULT current_setting('ai.openrouter_api_key', true)
) RETURNS vector
LANGUAGE plpython3u
AS $$
import json, urllib.request
url = 'https://openrouter.ai/api/v1/embeddings'
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
body = json.dumps({'model': model_name, 'input': input_text}).encode()
req = urllib.request.Request(url, data=body, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
return result['data'][0]['embedding']
$$;

-- Convenience: OpenRouter chat completion (uses pgai openai_chat_complete under the hood)
CREATE OR REPLACE FUNCTION public.openrouter_complete(
    model_name text,
    messages jsonb,
    api_key_name text DEFAULT 'OPENROUTER_API_KEY'
) RETURNS text
LANGUAGE sql
AS $$
    SELECT ai.openai_chat_complete(
        model_name,
        messages,
        api_key_name => api_key_name,
        client_config => ai.openai_client_config(base_url => 'https://openrouter.ai/api/v1')
    )->'choices'->0->'message'->>'content';
$$;

-- Generate summary from raw_text using openrouter_complete
CREATE OR REPLACE FUNCTION public.generate_summary(row_id INT)
RETURNS VOID AS $$
DECLARE
    rx_record RECORD;
    prompt TEXT;
    summary_text TEXT;
BEGIN
    SELECT * INTO rx_record FROM prescriptions WHERE id = row_id;
    IF rx_record.raw_text IS NULL OR rx_record.raw_text = '' THEN RETURN; END IF;

    prompt := 'Summarize the following prescription text in 3-5 sentences covering: '
           || 'patient info, diagnosis, key medications with dosages, and special instructions. '
           || 'Do not use markdown. Prescription text: ' || rx_record.raw_text;

    SELECT openrouter_complete(
        current_setting('ai.openrouter_complete_model'),
        jsonb_build_array(jsonb_build_object('role', 'user', 'content', prompt))
    ) INTO summary_text;

    UPDATE prescriptions SET summary = summary_text WHERE id = row_id;
END;
$$ LANGUAGE plpgsql;

-- Generate embedding of summary using openrouter_embed
CREATE OR REPLACE FUNCTION public.generate_embedding(row_id INT)
RETURNS VOID AS $$
DECLARE
    rx_record RECORD;
BEGIN
    SELECT * INTO rx_record FROM prescriptions WHERE id = row_id;
    IF rx_record.summary IS NULL OR rx_record.summary = '' THEN RETURN; END IF;

    DELETE FROM prescription_embeddings WHERE prescription_id = row_id;
    INSERT INTO prescription_embeddings (prescription_id, chunk_text, embedding)
    VALUES (row_id, rx_record.summary,
            openrouter_embed(current_setting('ai.openrouter_embed_model'), rx_record.summary));
END;
$$ LANGUAGE plpgsql;

-- Trigger: auto-generate summary on INSERT (swallows errors for rate-limits)
CREATE OR REPLACE FUNCTION public.trigger_generate_summary()
RETURNS TRIGGER AS $$
BEGIN
    BEGIN
        PERFORM generate_summary(NEW.id);
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'generate_summary failed for id %: %', NEW.id, SQLERRM;
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS after_insert_generate_summary ON prescriptions;
CREATE TRIGGER after_insert_generate_summary
AFTER INSERT ON prescriptions
FOR EACH ROW EXECUTE FUNCTION trigger_generate_summary();

-- Trigger: auto-generate embedding when summary is written/updated
CREATE OR REPLACE FUNCTION public.trigger_generate_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.summary IS NOT NULL AND NEW.summary != ''
       AND (OLD.summary IS NULL OR OLD.summary != NEW.summary) THEN
        BEGIN
            PERFORM generate_embedding(NEW.id);
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'generate_embedding failed for id %: %', NEW.id, SQLERRM;
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS after_update_generate_embedding ON prescriptions;
CREATE TRIGGER after_update_generate_embedding
AFTER UPDATE OF summary ON prescriptions
FOR EACH ROW EXECUTE FUNCTION trigger_generate_embedding();
