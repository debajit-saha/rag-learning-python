import os
import hashlib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
)
from azure.search.documents.models import VectorizedQuery

from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)


load_dotenv()

PDF_PATH = Path(__file__).resolve().parent / "employee_handbook.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

CONTAINER_NAME = os.environ["AZURE_STORAGE_CONTAINER"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX"]

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]

credential = AzureKeyCredential(SEARCH_KEY)

model = SentenceTransformer(MODEL_NAME)


# --------------------------------
# 1. Upload original PDF to Blob
# --------------------------------
def upload_pdf_to_blob():
    connection_string = os.environ[
        "AZURE_STORAGE_CONNECTION_STRING"
    ]

    blob_service = BlobServiceClient.from_connection_string(
        connection_string
    )

    container = blob_service.get_container_client(
        CONTAINER_NAME
    )

    if not container.exists():
        container.create_container()

    blob_client = container.get_blob_client(
        PDF_PATH.name
    )

    with PDF_PATH.open("rb") as file:
        blob_client.upload_blob(file, overwrite=True)

    print(f"Uploaded PDF to Blob: {PDF_PATH.name}")


# --------------------------------
# 2. Extract and chunk PDF
# --------------------------------
def chunk_text(text, size=1000, overlap=200):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunks():
    reader = PdfReader(str(PDF_PATH))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for chunk_number, content in enumerate(
            chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP),
            start=1
        ):
            raw_id = (
                f"{PDF_PATH.name}-{page_number}-{chunk_number}"
            )

            # Azure Search document keys use a safe identifier.
            document_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()

            records.append({
                "id": document_id,
                "file_name": PDF_PATH.name,
                "page_number": page_number,
                "chunk_number": chunk_number,
                "content": content,
            })

    return records


# --------------------------------
# 3. Create the vector index
# --------------------------------
def create_search_index():
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=credential
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="file_name",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SimpleField(
            name="chunk_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    metric="cosine"
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="handbook-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[
                SemanticField(field_name="content")
            ]
        ),
    )

    semantic_search = SemanticSearch(
        configurations=[semantic_config]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    index_client.create_or_update_index(index)
    print(f"Search index ready: {INDEX_NAME}")


# --------------------------------
# 4. Generate vectors and upload
# --------------------------------
def index_chunks(records):
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    contents = [record["content"] for record in records]

    vectors = model.encode(
        contents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    documents = []

    for record, vector in zip(records, vectors):
        documents.append({
            **record,
            "contentVector": vector.tolist(),
        })

    # Upload in batches.
    for start in range(0, len(documents), 100):
        batch = documents[start:start + 100]
        results = search_client.upload_documents(
            documents=batch
        )

        failures = [
            result for result in results
            if not result.succeeded
        ]

        if failures:
            raise RuntimeError(
                f"Index upload failed: {failures}"
            )

    print(f"Indexed {len(documents)} chunks.")


# --------------------------------
# 5. Hybrid search
# --------------------------------
def search_documents(question):
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vector_query = VectorizedQuery(
        vector=query_vector.tolist(),
        k_nearest_neighbors=TOP_K,
        fields="contentVector",
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="handbook-semantic-config",
        top=TOP_K,
        filter="file_name eq 'employee_handbook.pdf'",
        select=[
            "file_name",
            "page_number",
            "chunk_number",
            "content",
        ],
    )

    sources = []

    for rank, result in enumerate(results, start=1):
        reranker_score = result.get("@search.reranker_score")

        if reranker_score is not None and reranker_score < 2.0:
            print(
                f"Potentially irrelevant result "
                f"(semantic score: {reranker_score:.4f})"
            )
            return

        source = {
            "rank": rank,
            "file_name": result["file_name"],
            "page_number": result["page_number"],
            "chunk_number": result["chunk_number"],
            "content": result["content"],
            "search_score": result["@search.score"],
            "semantic_score": reranker_score
        }

        sources.append(source)

        print(f"\n--- Result {rank} ---")
        print(f"Source: {source['file_name']}, "
              f"Page {source['page_number']}")
        print(f"Chunk: {source['chunk_number']}")
        print(f"Content: {source['content']}")
        print(f"Search score: {source['search_score']:.4f}")
        print(f"Semantic score: {reranker_score:.4f}")

    return sources


# --------------------------------
# 6. Run the pipeline
# --------------------------------
def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Place {PDF_PATH.name} beside this script."
        )

    upload_pdf_to_blob()

    records = build_chunks()
    if not records:
        raise ValueError("No extractable PDF text found.")

    create_search_index()
    index_chunks(records)

    print("\nSearch is ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if question:
            search_documents(question)


if __name__ == "__main__":
    main()
