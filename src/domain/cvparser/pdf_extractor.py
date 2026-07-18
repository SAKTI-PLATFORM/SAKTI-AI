"""Server-side text extraction for uploaded PDF CVs."""

from __future__ import annotations

from io import BytesIO

import pdfplumber

MAX_CV_FILE_SIZE = 10 * 1024 * 1024


def extract_pdf_text(content: bytes) -> str:
    if not content:
        raise ValueError("File CV kosong.")
    if len(content) > MAX_CV_FILE_SIZE:
        raise ValueError("Ukuran CV melebihi batas maksimal 10 MB.")
    if b"%PDF-" not in content[:1024]:
        raise ValueError("File CV bukan PDF yang valid.")

    try:
        with pdfplumber.open(BytesIO(content)) as document:
            pages = [page.extract_text() or "" for page in document.pages]
    except Exception as exc:
        raise ValueError("File PDF rusak atau tidak dapat dibaca.") from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
    if not text:
        raise ValueError(
            "Teks CV tidak terdeteksi. Gunakan PDF berbasis teks, bukan hasil scan."
        )

    return text
