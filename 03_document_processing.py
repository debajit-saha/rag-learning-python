from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(__file__).resolve().parent / "employee_handbook.pdf"


def chunk_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


reader = PdfReader(pdf_path)

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""    

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200
    )

    for chunk_number, chunk in enumerate(chunks, start=1):
        print({
            "page_number": page_number,
            "chunk_number": chunk_number,
            "content": chunk
        })
