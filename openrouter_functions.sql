-- =============================================================================
-- openrouter_functions.sql — Custom PostgreSQL functions for OpenRouter API
--
-- 1. openrouter_embed  — Calls OpenRouter /v1/embeddings directly via HTTP
--                        (LiteLLM doesn't support OpenRouter as an embedding provider)
--
-- 2. openrouter_complete — Wraps ai.openai_chat_complete with OpenRouter base URL
--                          (pgai has no ai.litellm_chat_complete function)
--
-- Run: psql -h localhost -U postgres -f openrouter_functions.sql
-- =============================================================================

-- OpenRouter config (api key, model names) is injected by init_config.sh from env vars.
-- DB settings used: ai.openrouter_api_key, ai.openrouter_embed_model, ai.openrouter_complete_model

-- =============================================================================
-- 1. openrouter_embed(model_name, input_text, api_key)
--
-- Why this exists:
--   ai.litellm_embed('openrouter/...', ...) fails because LiteLLM's Python
--   library doesn't map OpenRouter as an embedding provider — only for chat.
--   ai.openai_embed(..., client_config => ...) also fails because OpenRouter
--   returns { "object": "list" } instead of OpenAI's expected format,
--   causing the response parser to crash.
--
-- This function calls OpenRouter's /v1/embeddings endpoint directly via
-- urllib, bypassing both LiteLLM and the OpenAI client.
--
-- Usage:
--   SELECT openrouter_embed('nvidia/llama-nemotron-embed-vl-1b-v2:free', 'hello world');
--   SELECT vector_dims(openrouter_embed('nvidia/llama-nemotron-embed-vl-1b-v2:free', 'test'));
-- =============================================================================
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

-- =============================================================================
-- 2. openrouter_complete(model_name, messages, api_key_name)
--
-- Why this exists:
--   pgai has no ai.litellm_chat_complete function — LiteLLM integration is
--   embedding-only. But pgai does have ai.openai_chat_complete which accepts
--   a client_config parameter to override the base URL.
--
--   Since OpenRouter exposes an OpenAI-compatible /v1/chat/completions API,
--   we just point ai.openai_chat_complete at https://openrouter.ai/api/v1.
--
--   This wrapper extracts the response text so you get a plain string back
--   instead of raw JSONB.
--
-- Usage:
--   SELECT openrouter_complete(
--       'google/gemma-4-31b-it:free',
--       jsonb_build_array(
--           jsonb_build_object('role', 'user', 'content', 'What is aspirin?')
--       )
--   );
-- =============================================================================
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


-- =============================================================================
-- 3. generate_summary(row_id INT)
--
-- Reads raw_text from a prescriptions row, calls openrouter_complete to
-- produce a concise clinical summary, and stores it in the summary column.
--
-- Usage:
--   SELECT generate_summary(1);
-- =============================================================================
CREATE OR REPLACE FUNCTION public.generate_summary(row_id INT)
RETURNS VOID AS $$
DECLARE
    rx_record RECORD;
    prompt TEXT;
    summary_text TEXT;
BEGIN
    SELECT * INTO rx_record
    FROM prescriptions
    WHERE id = row_id;

    IF rx_record.raw_text IS NULL OR rx_record.raw_text = '' THEN
        RETURN;
    END IF;

    prompt := 'Summarize the following prescription text in 3-5 sentences covering: '
           || 'patient info, diagnosis, key medications with dosages, and special instructions. '
           || 'Do not use markdown. Prescription text: ' || rx_record.raw_text;

    SELECT openrouter_complete(
        current_setting('ai.openrouter_complete_model'),
        jsonb_build_array(
            jsonb_build_object('role', 'user', 'content', prompt)
        )
    ) INTO summary_text;

    UPDATE prescriptions
    SET summary = summary_text
    WHERE id = row_id;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- 4. generate_embedding(row_id INT)
--
-- Reads the summary from a prescriptions row, calls openrouter_embed to
-- produce a 2048-dim vector, and upserts it into prescription_embeddings.
--
-- Usage:
--   SELECT generate_embedding(1);
-- =============================================================================
CREATE OR REPLACE FUNCTION public.generate_embedding(row_id INT)
RETURNS VOID AS $$
DECLARE
    rx_record RECORD;
BEGIN
    SELECT * INTO rx_record
    FROM prescriptions
    WHERE id = row_id;

    IF rx_record.summary IS NULL OR rx_record.summary = '' THEN
        RETURN;
    END IF;

    -- Remove old embedding for this prescription (upsert behaviour)
    DELETE FROM prescription_embeddings WHERE prescription_id = row_id;

    INSERT INTO prescription_embeddings (prescription_id, chunk_text, embedding)
    VALUES (
        row_id,
        rx_record.summary,
        openrouter_embed(current_setting('ai.openrouter_embed_model'), rx_record.summary)
    );
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- 5. Triggers
--
-- Chain: INSERT prescriptions → generate_summary → UPDATE summary
--        → generate_embedding → INSERT prescription_embeddings
--
-- Both triggers swallow errors so a rate-limit doesn't block the INSERT.
-- =============================================================================

-- Trigger function: auto-generate summary after INSERT
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
FOR EACH ROW
EXECUTE FUNCTION trigger_generate_summary();


-- Trigger function: auto-generate embedding after summary is written
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
FOR EACH ROW
EXECUTE FUNCTION trigger_generate_embedding();
