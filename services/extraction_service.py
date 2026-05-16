"""
Text & Image Extraction Service
- Extract text from PDFs using PyMuPDF
- Extract embedded images from PDFs
- OCR images using PaddleOCR
"""

import os
import io
import tempfile

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR

# Initialize PaddleOCR once (English, CPU for portability)
_ocr_engine = None


def _get_ocr_engine() -> PaddleOCR:
    """Lazy-init PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_engine


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text and images from a PDF using PyMuPDF.

    Returns:
        dict with keys:
            text        – full concatenated text from all pages
            page_texts  – list of per-page text
            images      – list of dicts {page, image_path} for extracted images
    """
    doc = fitz.open(pdf_path)
    page_texts: list[str] = []
    images: list[dict] = []
    tmp_dir = tempfile.mkdtemp(prefix="rx_images_")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # --- Text ---
        page_text = page.get_text("text")
        page_texts.append(page_text)

        # --- Embedded images ---
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_path = os.path.join(tmp_dir, f"page{page_num}_img{img_idx}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            images.append({"page": page_num, "image_path": img_path})

    doc.close()

    full_text = "\n\n".join(page_texts)
    return {"text": full_text, "page_texts": page_texts, "images": images}


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(image_path: str) -> str:
    """
    Run PaddleOCR on a single image file.

    Returns:
        Concatenated recognized text.
    """
    engine = _get_ocr_engine()
    results = engine.ocr(image_path)

    lines: list[str] = []
    if results and results[0]:
        for line in results[0]:
            # Handle both old format [(box, (text, conf))] and new format
            if isinstance(line, dict):
                lines.append(line.get("text", "") or line.get("rec_text", ""))
            else:
                text = line[1][0]  # (text, confidence)
                lines.append(text)

    return "\n".join(lines)


def ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR from raw image bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(img_bytes)
        tmp.close()
        return ocr_image(tmp.name)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Unified Extraction
# ---------------------------------------------------------------------------

def extract_from_file(file_path: str) -> dict:
    """
    Unified extraction entry point.

    For PDFs:
        1. Extract text via PyMuPDF.
        2. If text is too short (likely scanned), OCR each page as an image.
        3. Also OCR any embedded images.
    For images:
        1. Run PaddleOCR directly.

    Returns:
        dict with keys: text, ocr_texts, source
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        result = extract_text_from_pdf(file_path)
        main_text = result["text"].strip()

        ocr_texts: list[str] = []

        # If the PDF text is very short, it's likely a scanned document → OCR pages
        if len(main_text) < 50:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                page_ocr = ocr_image_bytes(img_bytes)
                ocr_texts.append(page_ocr)
            doc.close()
            main_text = "\n\n".join(ocr_texts) if ocr_texts else main_text

        # Also OCR embedded images for any extra handwritten notes
        for img_info in result["images"]:
            img_ocr = ocr_image(img_info["image_path"])
            if img_ocr.strip():
                ocr_texts.append(img_ocr)

        combined = main_text
        if ocr_texts:
            combined = main_text + "\n\n--- OCR from images ---\n\n" + "\n\n".join(ocr_texts)

        return {"text": combined, "ocr_texts": ocr_texts, "source": "pdf"}

    else:
        # Direct image
        ocr_text = ocr_image(file_path)
        return {"text": ocr_text, "ocr_texts": [ocr_text], "source": "image"}
