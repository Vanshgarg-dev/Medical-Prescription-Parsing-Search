"""
Supabase Storage Service
- Upload prescription files (PDF/Image) to Supabase Storage
- Download files from Supabase Storage
- Generate public/signed URLs
"""

import os
import uuid

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "prescriptions")

_client: Client | None = None


def _get_client() -> Client:
    """Initialize Supabase client (only once)."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _ensure_bucket():
    """Create the storage bucket if it doesn't exist."""
    client = _get_client()
    try:
        client.storage.get_bucket(SUPABASE_BUCKET)
    except Exception:
        client.storage.create_bucket(
            SUPABASE_BUCKET,
            options={"public": True},
        )


def upload_file(local_path: str, destination_folder: str = "prescriptions") -> dict:
    """
    Upload a local file to Supabase Storage.

    Args:
        local_path: Absolute path to the local file.
        destination_folder: Folder inside the bucket.

    Returns:
        dict with keys: storage_path, public_url, file_name, file_type
    """
    _ensure_bucket()
    client = _get_client()

    file_name = os.path.basename(local_path)
    unique_name = f"{uuid.uuid4().hex}_{file_name}"
    storage_path = f"{destination_folder}/{unique_name}"

    # Detect content type
    ext = os.path.splitext(file_name)[1].lower()
    content_type_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    with open(local_path, "rb") as f:
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": content_type},
        )

    public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)

    return {
        "storage_path": storage_path,
        "public_url": public_url,
        "file_name": file_name,
        "file_type": "pdf" if ext == ".pdf" else "image",
    }


def download_file(storage_path: str, local_dest: str) -> str:
    """
    Download a file from Supabase Storage to a local path.

    Args:
        storage_path: Path inside the bucket.
        local_dest: Local destination path.

    Returns:
        local_dest path.
    """
    client = _get_client()
    data = client.storage.from_(SUPABASE_BUCKET).download(storage_path)
    with open(local_dest, "wb") as f:
        f.write(data)
    return local_dest


def get_signed_url(storage_path: str, expiry_seconds: int = 86400) -> str:
    """Generate a signed URL for temporary access (default 24h)."""
    client = _get_client()
    result = client.storage.from_(SUPABASE_BUCKET).create_signed_url(
        storage_path, expiry_seconds
    )
    return result["signedURL"]


def list_files(prefix: str = "prescriptions/") -> list[dict]:
    """List all files under a given prefix."""
    client = _get_client()
    files = client.storage.from_(SUPABASE_BUCKET).list(prefix)
    return [
        {"name": f["name"], "size": f.get("metadata", {}).get("size"), "updated": f.get("updated_at")}
        for f in files
    ]
