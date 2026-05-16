"""
LLM Parsing Service (Groq)
- Parse prescription text into structured medical entities using Groq
- Uses tool calling for structured extraction
- Uses Postgres schema introspection to align output with DB schema
"""

import os
import json

from groq import Groq
from dotenv import load_dotenv

from services.database_service import introspect_schema

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Tool definitions for Groq function calling
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_prescription_entities",
            "description": (
                "Store parsed medical entities extracted from a prescription. "
                "Call this once you have extracted all information from the text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient",
                    },
                    "patient_age": {
                        "type": "string",
                        "description": "Age of the patient (e.g. '45 years')",
                    },
                    "patient_gender": {
                        "type": "string",
                        "description": "Gender of the patient",
                    },
                    "doctor_name": {
                        "type": "string",
                        "description": "Name of the prescribing doctor",
                    },
                    "doctor_speciality": {
                        "type": "string",
                        "description": "Speciality/department of the doctor",
                    },
                    "hospital_clinic": {
                        "type": "string",
                        "description": "Hospital or clinic name",
                    },
                    "diagnosis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of diagnoses mentioned",
                    },
                    "symptoms": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "List of symptoms mentioned",
                    },
                    "medications": {
                        "type": "array",
                        "description": "List of prescribed medications",
                        "items": {
                            "type": "object",
                            "properties": {
                                "drug_name": {
                                    "type": "string",
                                    "description": "Name of the drug/medication",
                                },
                                "dosage": {
                                    "type": ["string", "null"],
                                    "description": "Dosage (e.g. '500mg')",
                                },
                                "frequency": {
                                    "type": ["string", "null"],
                                    "description": "How often (e.g. 'twice daily')",
                                },
                                "duration": {
                                    "type": ["string", "null"],
                                    "description": "Duration (e.g. '7 days')",
                                },
                                "route": {
                                    "type": ["string", "null"],
                                    "description": "Route of administration (e.g. 'oral', 'IV')",
                                },
                                "instructions": {
                                    "type": ["string", "null"],
                                    "description": "Special instructions (e.g. 'after meals')",
                                },
                            },
                            "required": ["drug_name"],
                        },
                    },
                },
                "required": ["medications"],
            },
        },
    }
]


def _build_system_prompt() -> str:
    """
    Build system prompt that includes the live PostgreSQL schema
    so the LLM knows exactly what columns/types to fill.
    """
    schema_info = introspect_schema()

    return f"""You are a medical prescription parser. Your job is to extract structured medical information from prescription text.

## Database Schema (PostgreSQL)
The extracted data will be stored in the following tables. Align your output precisely with these columns:

{schema_info}

## Instructions
1. Read the prescription text carefully.
2. Extract ALL medical entities: patient info, doctor info, diagnoses, symptoms, and medications.
3. Call the `store_prescription_entities` tool with the extracted data.
4. If a field is not present in the text, omit it or use null.
5. For medications, extract every prescribed drug with as much detail as available.
6. Be precise — do not hallucinate information not present in the text."""


def parse_prescription(raw_text: str) -> dict:
    """
    Parse raw prescription text into structured entities using Groq with tool calling.

    Args:
        raw_text: The extracted text from the prescription document.

    Returns:
        dict of parsed medical entities matching the DB schema.
    """
    system_prompt = _build_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Parse the following prescription text and call the store_prescription_entities tool:\n\n---\n{raw_text}\n---",
        },
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "store_prescription_entities"}},
        temperature=0.0,
        max_tokens=4096,
    )

    message = response.choices[0].message

    # Extract tool call arguments
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        parsed = json.loads(tool_call.function.arguments)
        return parsed

    # Fallback: try to parse from content
    if message.content:
        try:
            return json.loads(message.content)
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse prescription", "raw_response": message.content}


def semantic_search_query(user_query: str) -> str:
    """
    Use Groq to transform a natural language search query into a SQL WHERE clause
    for searching prescriptions. Uses schema introspection.

    Args:
        user_query: Natural language query (e.g. "find patients on metformin")

    Returns:
        SQL WHERE clause string.
    """
    schema_info = introspect_schema()

    messages = [
        {
            "role": "system",
            "content": f"""You are a SQL query builder for a medical prescriptions database.

## Database Schema
{schema_info}

## Instructions
Given a natural language query, produce a SQL query that searches across the prescriptions,
medical_entities, and medications tables.
Return ONLY the full SQL SELECT query, nothing else. Use ILIKE for text matching.
Always join the tables properly using foreign keys.
Limit results to 20 rows.""",
        },
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
