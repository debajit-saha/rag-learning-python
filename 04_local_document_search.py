from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
PDF_PATH = Path(__file__).resolve().parent / "employee_handbook.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# 1. Extract text from PDF
# -----------------------------
def extract_pdf_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


# -----------------------------
# 2. Split text into chunks
# -----------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

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


# -----------------------------
# 3. Build document collection
# -----------------------------
def build_collection(pdf_path):
    pages = extract_pdf_pages(pdf_path)
    collection = []

    for page in pages:
        chunks = chunk_text(
            page["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(chunks, start=1):
            collection.append({
                "id": f"page-{page['page_number']}-chunk-{chunk_number}",
                "file_name": pdf_path.name,
                "page_number": page["page_number"],
                "chunk_number": chunk_number,
                "content": chunk
            })

    return collection


# -----------------------------
# 4. Generate embeddings
# -----------------------------
def embed_documents(collection, model):
    texts = [item["content"] for item in collection]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


# -----------------------------
# 5. Search by semantic similarity
# -----------------------------
def search(query, collection, embeddings, model, top_k=3):
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Normalized vectors: dot product equals cosine similarity.
    scores = embeddings @ query_vector

    # Highest similarity first.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = collection[int(index)].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


# -----------------------------
# 6. Run the application
# -----------------------------
def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Extracting and chunking PDF...")
    collection = build_collection(PDF_PATH)

    if not collection:
        print("No extractable text found in the PDF.")
        return

    print("Generating document embeddings...")
    embeddings = embed_documents(collection, model)

    print(f"\nDocuments/chunks: {len(collection)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    print("\nSemantic search is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = search(
            query,
            collection,
            embeddings,
            model,
            TOP_K
        )

        print("\nTop matching chunks:")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Source: {result['file_name']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk ID: {result['id']}")
            print(f"Content:\n{result['content']}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
