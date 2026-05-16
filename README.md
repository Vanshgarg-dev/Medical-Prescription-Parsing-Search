# Medical Prescription Parsing & Search

Automated system to upload, extract, parse, and search medical prescriptions.

## Architecture

```
Upload prescription (PDF/Image)
       ↓
Store securely in Supabase Storage
       ↓
Extract text using PyMuPDF + PaddleOCR
       ↓
Parse medical entities using Groq LLM (tool calling)
       ↓
Store structured data in PostgreSQL (TimescaleDB + pgvector)
       ↓
Auto-generate summary (OpenRouter) ← DB trigger
       ↓
Auto-generate embedding (OpenRouter) ← DB trigger
       ↓
Enable semantic, full-text, and deep search
```

## Project Structure

```
├── .env                          # All credentials (not committed)
├── .env.example                  # Template for .env
├── .gitignore
├── compose.yaml                  # Docker Compose for PostgreSQL + pgai vectorizer
├── init_db.sql                   # Schema, functions, and triggers
├── openrouter_functions.sql      # OpenRouter embed/complete SQL functions
├── app.py                        # Gradio web UI
├── main.py                       # CLI orchestrator
├── requirements.txt              # Python dependencies
└── services/
    ├── supabase_service.py       # Supabase Storage upload/download
    ├── extraction_service.py     # PyMuPDF text extraction + PaddleOCR
    ├── llm_service.py            # Groq LLM parsing with tool calling
    └── database_service.py       # PostgreSQL CRUD + schema introspection + search
```

## Setup

### 1. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- **`GROQ_API_KEY`** — LLM parsing and natural-language search
- **`OPENROUTER_API_KEY`** — summary generation and embeddings (via PostgreSQL triggers)
- **`SUPABASE_URL`**, **`SUPABASE_KEY`** — file storage
- **`POSTGRES_PASSWORD`** — database password

### 2. Start PostgreSQL + Vectorizer

```bash
docker compose up -d
```

This starts:
- **TimescaleDB** (PostgreSQL 17 + pgvector + pgai + plpython3u)
- **pgai vectorizer-worker** for background embedding jobs

On first run, `init_db.sql` creates the schema, OpenRouter SQL functions, and DB triggers that auto-generate summaries and embeddings on every prescription insert.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Web UI (Gradio)

```bash
python app.py
```

Opens at `http://localhost:7860` with tabs for upload, search, and prescription details.

### CLI — Upload & Process

```bash
python main.py upload /path/to/prescription.pdf
python main.py upload /path/to/prescription.jpg
```

Pipeline steps:
1. Upload file to Supabase Storage
2. Extract text (PyMuPDF for PDFs, PaddleOCR for images/scanned PDFs)
3. Parse medical entities via Groq LLM with tool calling
4. Store structured data in PostgreSQL
5. Auto-generate AI summary (DB trigger → OpenRouter)
6. Auto-generate vector embedding (DB trigger → OpenRouter)

### CLI — Search

```bash
# Natural language search (LLM generates SQL using schema introspection)
python main.py search "find all patients prescribed metformin in the last month"

# Semantic search on summaries (vector similarity)
python main.py similar "diabetes medication with insulin"

# Direct searches
python main.py drug "amoxicillin"
python main.py patient "John Doe"
python main.py diagnosis "diabetes"
python main.py fulltext "hypertension treatment"

# Get full prescription details
python main.py get 1
```

### Inspect DB Schema

```bash
python main.py schema
```

## Key Features

- **Groq LLM + Tool Calling**: Structured extraction via forced tool calls matching the DB schema
- **OpenRouter Triggers**: DB triggers auto-generate summaries and vector embeddings on insert using OpenRouter API
- **Semantic Search**: pgvector cosine similarity search over prescription summary embeddings
- **Schema Introspection**: The LLM receives the live PostgreSQL schema so output always aligns with actual columns
- **PaddleOCR**: Handles scanned PDFs and image prescriptions
- **Deep Search**: Natural language queries are translated to SQL via Groq
- **Trigram Search**: Fast fuzzy matching on drug names, patient names, doctor names
- **Full-Text Search**: PostgreSQL `tsvector` search on raw prescription text
- **Gradio UI**: Web interface for upload, search, and prescription viewing
