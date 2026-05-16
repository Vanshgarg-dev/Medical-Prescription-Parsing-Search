# Medical Prescription Parsing & Search

Upload prescriptions (PDF or image), extract structured medical data, and run various searches over it.

## How it works

```
PDF/Image → Supabase Storage → PyMuPDF + PaddleOCR → Groq LLM → PostgreSQL
                                                                      ↓
                                                          DB triggers generate
                                                        summary + embedding via
                                                              OpenRouter
```

On insert, database triggers call OpenRouter to produce a plain-text summary and a vector embedding. These power the semantic search.

## Project structure

```
.env.example
compose.yaml
init_db.sql
openrouter_functions.sql
app.py
main.py
requirements.txt
services/
  supabase_service.py
  extraction_service.py
  llm_service.py
  database_service.py
```

## Setup

```bash
cp .env.example .env
```

Fill in your `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, and `POSTGRES_PASSWORD`.

Start the database:

```bash
docker compose up -d
```

This brings up TimescaleDB (pg17 + pgvector + pgai) and a vectorizer worker. On first boot `init_db.sql` sets up the schema, functions, and triggers.

Install Python deps:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Runs on `http://localhost:7860`.

## Stack

- **Groq** — LLM extraction via tool calling, natural-language-to-SQL search
- **OpenRouter** — summary generation and embeddings (called from PL/pgSQL triggers)
- **PostgreSQL + pgvector** — structured storage, trigram indexes, full-text search, vector similarity
- **PaddleOCR** — OCR for scanned documents and images
- **Supabase** — file storage
- **Gradio** — web frontend
