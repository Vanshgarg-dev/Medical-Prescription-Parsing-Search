"""
PostgreSQL Database Service
- Schema introspection for LLM context
- Store prescriptions, entities, and medications
- Full-text, trigram, and semantic search
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema Introspection
# ---------------------------------------------------------------------------

def introspect_schema() -> str:
    """
    Introspect the PostgreSQL schema and return a human-readable description
    of all tables, columns, types, and constraints. This is fed to the LLM
    so it can produce output aligned with the actual DB structure.
    """
    query = """
    SELECT
        t.table_name,
        c.column_name,
        c.data_type,
        c.udt_name,
        c.is_nullable,
        c.column_default,
        tc.constraint_type,
        ccu.table_name AS referenced_table
    FROM information_schema.tables t
    JOIN information_schema.columns c
        ON t.table_name = c.table_name AND t.table_schema = c.table_schema
    LEFT JOIN information_schema.key_column_usage kcu
        ON c.column_name = kcu.column_name AND c.table_name = kcu.table_name
           AND c.table_schema = kcu.table_schema
    LEFT JOIN information_schema.table_constraints tc
        ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
    LEFT JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
           AND tc.constraint_type = 'FOREIGN KEY'
    WHERE t.table_schema = 'public'
      AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name, c.ordinal_position;
    """

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except Exception:
        # Fallback if DB not available (e.g. during import)
        return _fallback_schema()

    if not rows:
        return _fallback_schema()

    # Group by table
    tables: dict[str, list] = {}
    for row in rows:
        table = row["table_name"]
        tables.setdefault(table, []).append(row)

    lines = []
    for table, cols in tables.items():
        lines.append(f"### Table: {table}")
        seen = set()
        for col in cols:
            key = (col["column_name"], col["data_type"])
            if key in seen:
                continue
            seen.add(key)

            nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
            constraint = ""
            if col.get("constraint_type") == "PRIMARY KEY":
                constraint = " [PK]"
            elif col.get("constraint_type") == "FOREIGN KEY":
                constraint = f" [FK → {col.get('referenced_table', '?')}]"

            dtype = col["udt_name"] if col["udt_name"] != col["data_type"] else col["data_type"]
            lines.append(f"  - {col['column_name']}: {dtype} {nullable}{constraint}")
        lines.append("")

    return "\n".join(lines)


def _fallback_schema() -> str:
    """Hardcoded fallback schema description if DB is unreachable."""
    return """### Table: prescriptions
  - id: serial NOT NULL [PK]
  - file_name: text NOT NULL
  - storage_url: text NOT NULL
  - file_type: text NOT NULL
  - raw_text: text NULL
  - summary: text NULL
  - uploaded_at: timestamptz NULL

### Table: medical_entities
  - id: serial NOT NULL [PK]
  - prescription_id: int4 NOT NULL [FK → prescriptions]
  - patient_name: text NULL
  - patient_age: text NULL
  - patient_gender: text NULL
  - doctor_name: text NULL
  - doctor_speciality: text NULL
  - hospital_clinic: text NULL
  - diagnosis: _text NULL
  - symptoms: _text NULL
  - parsed_at: timestamptz NULL

### Table: medications
  - id: serial NOT NULL [PK]
  - entity_id: int4 NOT NULL [FK → medical_entities]
  - drug_name: text NOT NULL
  - dosage: text NULL
  - frequency: text NULL
  - duration: text NULL
  - route: text NULL
  - instructions: text NULL

### Table: prescription_embeddings
  - id: serial NOT NULL [PK]
  - prescription_id: int4 NOT NULL [FK → prescriptions]
  - chunk_text: text NOT NULL
  - embedding: vector(2048) NULL
"""


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------

def insert_prescription(file_name: str, storage_url: str, file_type: str, raw_text: str) -> int:
    """Insert a new prescription record. Returns the prescription id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO prescriptions (file_name, storage_url, file_type, raw_text)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (file_name, storage_url, file_type, raw_text),
            )
            return cur.fetchone()[0]


def insert_medical_entities(prescription_id: int, entities: dict) -> int:
    """Insert parsed medical entities. Returns the entity id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO medical_entities
                   (prescription_id, patient_name, patient_age, patient_gender,
                    doctor_name, doctor_speciality, hospital_clinic,
                    diagnosis, symptoms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    prescription_id,
                    entities.get("patient_name"),
                    entities.get("patient_age"),
                    entities.get("patient_gender"),
                    entities.get("doctor_name"),
                    entities.get("doctor_speciality"),
                    entities.get("hospital_clinic"),
                    entities.get("diagnosis"),
                    entities.get("symptoms"),
                ),
            )
            return cur.fetchone()[0]


def insert_medications(entity_id: int, medications: list[dict]) -> list[int]:
    """Insert medications for a medical entity. Returns list of medication ids."""
    ids = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for med in medications:
                cur.execute(
                    """INSERT INTO medications
                       (entity_id, drug_name, dosage, frequency, duration, route, instructions)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        entity_id,
                        med.get("drug_name", "Unknown"),
                        med.get("dosage"),
                        med.get("frequency"),
                        med.get("duration"),
                        med.get("route"),
                        med.get("instructions"),
                    ),
                )
                ids.append(cur.fetchone()[0])
    return ids


# ---------------------------------------------------------------------------
# Search / Queries
# ---------------------------------------------------------------------------

def search_by_drug(drug_name: str) -> list[dict]:
    """Search prescriptions by drug name (trigram similarity)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT p.id AS prescription_id, p.file_name, me.patient_name,
                          me.doctor_name, m.drug_name, m.dosage, m.frequency, m.duration
                   FROM medications m
                   JOIN medical_entities me ON m.entity_id = me.id
                   JOIN prescriptions p ON me.prescription_id = p.id
                   WHERE m.drug_name ILIKE %s
                   ORDER BY p.uploaded_at DESC
                   LIMIT 20""",
                (f"%{drug_name}%",),
            )
            return [dict(row) for row in cur.fetchall()]


def search_by_patient(patient_name: str) -> list[dict]:
    """Search prescriptions by patient name."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT p.id AS prescription_id, p.file_name, me.patient_name,
                          me.doctor_name, me.diagnosis, me.symptoms
                   FROM medical_entities me
                   JOIN prescriptions p ON me.prescription_id = p.id
                   WHERE me.patient_name ILIKE %s
                   ORDER BY p.uploaded_at DESC
                   LIMIT 20""",
                (f"%{patient_name}%",),
            )
            return [dict(row) for row in cur.fetchall()]


def search_by_diagnosis(diagnosis: str) -> list[dict]:
    """Search prescriptions by diagnosis."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT p.id AS prescription_id, p.file_name, me.patient_name,
                          me.doctor_name, me.diagnosis, m.drug_name, m.dosage
                   FROM medical_entities me
                   JOIN prescriptions p ON me.prescription_id = p.id
                   LEFT JOIN medications m ON m.entity_id = me.id
                   WHERE EXISTS (
                       SELECT 1 FROM unnest(me.diagnosis) d WHERE d ILIKE %s
                   )
                   ORDER BY p.uploaded_at DESC
                   LIMIT 20""",
                (f"%{diagnosis}%",),
            )
            return [dict(row) for row in cur.fetchall()]


def search_fulltext(query: str) -> list[dict]:
    """Full-text search across raw prescription text."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT p.id, p.file_name, p.uploaded_at,
                          ts_rank(to_tsvector('english', p.raw_text),
                                  plainto_tsquery('english', %s)) AS rank
                   FROM prescriptions p
                   WHERE to_tsvector('english', p.raw_text) @@ plainto_tsquery('english', %s)
                   ORDER BY rank DESC
                   LIMIT 20""",
                (query, query),
            )
            return [dict(row) for row in cur.fetchall()]


def execute_raw_query(sql: str) -> list[dict]:
    """Execute a raw SQL query (for LLM-generated queries). SELECT only."""
    sql_stripped = sql.strip().rstrip(";")
    if not sql_stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_stripped)
            return [dict(row) for row in cur.fetchall()]


def get_prescription_by_id(prescription_id: int) -> dict | None:
    """Get a single prescription with all its entities and medications."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Prescription
            cur.execute("SELECT * FROM prescriptions WHERE id = %s", (prescription_id,))
            rx = cur.fetchone()
            if not rx:
                return None
            result = dict(rx)

            # Entities
            cur.execute("SELECT * FROM medical_entities WHERE prescription_id = %s", (prescription_id,))
            entities = [dict(e) for e in cur.fetchall()]

            for entity in entities:
                cur.execute("SELECT * FROM medications WHERE entity_id = %s", (entity["id"],))
                entity["medications"] = [dict(m) for m in cur.fetchall()]

            result["entities"] = entities
            return result


# ---------------------------------------------------------------------------
# Summary & Embeddings (via DB-side OpenRouter functions)
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")


def get_prescription_summary(prescription_id: int) -> str:
    """Read back the auto-generated summary (populated by DB trigger on INSERT)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT summary FROM prescriptions WHERE id = %s", (prescription_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else ""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic (vector) search: embed the query, then find closest prescription
    summaries by cosine distance.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    pe.prescription_id,
                    p.file_name,
                    p.summary,
                    me.patient_name,
                    me.doctor_name,
                    me.diagnosis,
                    1 - (pe.embedding <=> openrouter_embed(%s, %s)) AS similarity
                FROM prescription_embeddings pe
                JOIN prescriptions p ON p.id = pe.prescription_id
                LEFT JOIN medical_entities me ON me.prescription_id = p.id
                ORDER BY pe.embedding <=> openrouter_embed(%s, %s)
                LIMIT %s;
                """,
                (EMBED_MODEL, query, EMBED_MODEL, query, top_k),
            )
            return [dict(row) for row in cur.fetchall()]
