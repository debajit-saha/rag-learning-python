"""RAG API for the employee handbook.

On startup it:
  1. Uploads employee_handbook.pdf to Blob Storage.
  2. Chunks the PDF text and indexes it into AZURE_SEARCH_INDEX.
  3. Extracts the PDF images, describes them with Azure OpenAI,
     uploads them to Blob Storage, and indexes the descriptions
     into AZURE_SEARCH_IMAGE_INDEX.

Ingestion is skipped when the PDF and embedding deployment haven't
changed since the last run. Delete ingestion_state.json to force a
full re-ingest.

Run with:  python app.py
"""

import base64
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pymupdf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import OpenAI
from urllib.parse import urlparse

load_dotenv()

# --------------------------------
# Configuration
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "employee_handbook.pdf"
IMAGE_DIR = BASE_DIR / "extracted_images"
DESCRIPTION_FILE = BASE_DIR / "image_descriptions.json"
STATE_FILE = BASE_DIR / "ingestion_state.json"

TEXT_INDEX = os.environ["AZURE_SEARCH_INDEX"]
IMAGE_INDEX = os.environ["AZURE_SEARCH_IMAGE_INDEX"]
TEXT_SEMANTIC_CONFIG = "handbook-semantic-config"
IMAGE_SEMANTIC_CONFIG = "image-semantic-config"

# Embedding model deployed in Foundry. The search index vectorizer uses
# the same deployment to embed queries (e.g. from the agent's search tool).
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
# Base model name; only needed if the deployment name differs from it.
EMBEDDING_MODEL = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL", EMBEDDING_DEPLOYMENT
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Minimum semantic reranker score for an image to be shown.
IMAGE_SCORE_THRESHOLD = 2.5

# Source markers the agent adds to its answer, e.g. 【6:0†source】.
CITATION_MARKER = re.compile(r"\s*【[^】]*】")

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}

# --------------------------------
# Azure clients
# --------------------------------
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=search_credential,
)

text_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=TEXT_INDEX,
    credential=search_credential,
)

image_search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=IMAGE_INDEX,
    credential=search_credential,
)

blob_service = BlobServiceClient.from_connection_string(
    os.environ["AZURE_STORAGE_CONNECTION_STRING"]
)

blob_container = blob_service.get_container_client(
    os.environ["AZURE_STORAGE_CONTAINER"]
)

openai_client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

openai_model = os.environ["AZURE_OPENAI_DEPLOYMENT"]

project = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent_client = project.get_openai_client(
    agent_name=os.environ["AGENT_NAME"]
)


# ================================
# Ingestion
# ================================
def upload_blob(name: str, data: bytes, content_type: str):
    blob_container.get_blob_client(name).upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def upload_documents(
    client: SearchClient, index_name: str, documents: list[dict]
):
    for start in range(0, len(documents), 100):
        results = client.upload_documents(
            documents=documents[start:start + 100]
        )
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(
                f"Failed to index {len(failed)} documents in {index_name}."
            )


# ---------- 1. PDF to Blob ----------
def upload_pdf(pdf_bytes: bytes):
    if not blob_container.exists():
        blob_container.create_container()

    upload_blob(PDF_PATH.name, pdf_bytes, "application/pdf")
    print(f"Uploaded PDF to Blob: {PDF_PATH.name}")


# ---------- 2. Text index ----------
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
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


def embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]


def vectorizer_resource_url() -> str:
    # The search vectorizer needs the resource root, e.g.
    # https://<name>.openai.azure.com, not the /openai/v1 path.
    host = urlparse(os.environ["AZURE_OPENAI_ENDPOINT"]).hostname
    name = host.split(".")[0]
    return f"https://{name}.openai.azure.com"


def create_text_index(dimensions: int):
    # Vector dimensions can't be changed in place, so recreate the
    # index if it was built with a different embedding model.
    try:
        existing = index_client.get_index(TEXT_INDEX)
    except ResourceNotFoundError:
        existing = None

    old_dimensions = next(
        (
            f.vector_search_dimensions
            for f in (existing.fields if existing else [])
            if f.name == "contentVector"
        ),
        None,
    )
    if existing and old_dimensions != dimensions:
        index_client.delete_index(TEXT_INDEX)
        print(
            f"Deleted {TEXT_INDEX}: vector size changed "
            f"{old_dimensions} -> {dimensions}"
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
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(metric="cosine"),
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="foundry-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=vectorizer_resource_url(),
                    deployment_name=EMBEDDING_DEPLOYMENT,
                    model_name=EMBEDDING_MODEL,
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
                vectorizer_name="foundry-vectorizer",
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=TEXT_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )

    index_client.create_or_update_index(
        SearchIndex(
            name=TEXT_INDEX,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
    )
    print(f"Text index ready: {TEXT_INDEX}")


def index_text(doc: pymupdf.Document):
    records = []

    for page_number, page in enumerate(doc, start=1):
        chunks = chunk_text(page.get_text(), CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_number, content in enumerate(chunks, start=1):
            raw_id = f"{PDF_PATH.name}-{page_number}-{chunk_number}"
            records.append({
                "id": hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
                "file_name": PDF_PATH.name,
                "page_number": page_number,
                "chunk_number": chunk_number,
                "content": content,
            })

    if not records:
        raise ValueError("No extractable text found in the PDF.")

    vectors = embed([r["content"] for r in records])

    create_text_index(dimensions=len(vectors[0]))

    # Remove old chunks so nothing stale remains if the PDF got shorter.
    old_ids = [
        {"id": r["id"]}
        for r in text_search_client.search(
            search_text="*",
            filter=f"file_name eq '{PDF_PATH.name}'",
            select=["id"],
        )
    ]
    if old_ids:
        text_search_client.delete_documents(documents=old_ids)

    documents = [
        {**record, "contentVector": vector}
        for record, vector in zip(records, vectors)
    ]

    upload_documents(text_search_client, TEXT_INDEX, documents)
    print(f"Indexed {len(documents)} text chunks into {TEXT_INDEX}.")


# ---------- 3. Images ----------
def extract_images(doc: pymupdf.Document) -> list[dict]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    images = []

    for page_number, page in enumerate(doc, start=1):
        for image_number, image_info in enumerate(
            page.get_images(full=True), start=1
        ):
            image = doc.extract_image(image_info[0])
            extension = image["ext"].lower()

            if extension not in MIME_TYPES:
                print(f"Skipping unsupported image type: {extension}")
                continue

            name = f"page_{page_number}_image_{image_number}.{extension}"
            data = image["image"]
            (IMAGE_DIR / name).write_bytes(data)

            images.append({
                "image_name": name,
                "page_number": page_number,
                "mime_type": MIME_TYPES[extension],
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
            })

    print(f"Extracted {len(images)} images from the PDF.")
    return images


def describe_image(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")

    response = openai_client.responses.create(
        model=openai_model,
        input=[
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "Describe images for a document search system. "
                        "Identify the main subject, visible text, "
                        "process steps, labels, and important details. "
                        "Do not invent details that are not visible."
                    ),
                }],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this image so someone can find it "
                            "using a natural-language question."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded}",
                    },
                ],
            },
        ],
        max_output_tokens=500,
    )

    return response.output_text


def describe_images(images: list[dict]):
    """Adds a description to each image, reusing cached descriptions."""
    cache = {}
    if DESCRIPTION_FILE.exists():
        for item in json.loads(DESCRIPTION_FILE.read_text(encoding="utf-8")):
            cache[item["image_name"]] = item

    for image in images:
        cached = cache.get(image["image_name"])

        # Older cache entries have no hash; trust them by name.
        if cached and cached.get("sha256", image["sha256"]) == image["sha256"]:
            image["description"] = cached["description"]
            continue

        print(f"Describing {image['image_name']}...")
        image["description"] = describe_image(image["data"], image["mime_type"])

    DESCRIPTION_FILE.write_text(
        json.dumps(
            [
                {
                    "image_name": image["image_name"],
                    "sha256": image["sha256"],
                    "description": image["description"],
                }
                for image in images
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def create_image_index():
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        SearchableField(
            name="image_name",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="description",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="blob_name",
            type=SearchFieldDataType.String,
        ),
    ]

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=IMAGE_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="image_name"),
                    content_fields=[SemanticField(field_name="description")],
                ),
            )
        ]
    )

    index_client.create_or_update_index(
        SearchIndex(
            name=IMAGE_INDEX,
            fields=fields,
            semantic_search=semantic_search,
        )
    )
    print(f"Image index ready: {IMAGE_INDEX}")


def index_images(doc: pymupdf.Document):
    images = extract_images(doc)
    if not images:
        print("No images found in the PDF.")
        return

    describe_images(images)

    for image in images:
        upload_blob(image["image_name"], image["data"], image["mime_type"])
    print(f"Uploaded {len(images)} images to Blob Storage.")

    create_image_index()

    documents = [
        {
            "id": hashlib.sha256(
                image["image_name"].encode("utf-8")
            ).hexdigest(),
            "image_name": image["image_name"],
            "description": image["description"],
            "page_number": image["page_number"],
            # Blob path inside AZURE_STORAGE_CONTAINER.
            "blob_name": image["image_name"],
        }
        for image in images
    ]

    # Remove entries from earlier runs that no longer match an image,
    # e.g. ones indexed under an older blob path.
    current_ids = {document["id"] for document in documents}
    stale_ids = [
        {"id": r["id"]}
        for r in image_search_client.search(search_text="*", select=["id"])
        if r["id"] not in current_ids
    ]
    if stale_ids:
        image_search_client.delete_documents(documents=stale_ids)
        print(f"Removed {len(stale_ids)} stale image entries.")

    upload_documents(image_search_client, IMAGE_INDEX, documents)
    print(f"Indexed {len(documents)} images into {IMAGE_INDEX}.")


# ---------- Run all ----------
def ingest():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf_bytes = PDF_PATH.read_bytes()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

    if (
        state.get("pdf_sha256") == pdf_hash
        and state.get("text_index") == TEXT_INDEX
        and state.get("image_index") == IMAGE_INDEX
        and state.get("embedding_deployment") == EMBEDDING_DEPLOYMENT
    ):
        print("PDF unchanged since last ingestion; skipping.")
        return

    print("Ingesting employee handbook...")
    upload_pdf(pdf_bytes)

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        index_text(doc)
        index_images(doc)

    STATE_FILE.write_text(
        json.dumps(
            {
                "pdf_sha256": pdf_hash,
                "text_index": TEXT_INDEX,
                "image_index": IMAGE_INDEX,
                "embedding_deployment": EMBEDDING_DEPLOYMENT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Ingestion complete.")


# ================================
# API
# ================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    ingest()
    yield


app = FastAPI(title="RAG Learning API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None


@app.get("/")
def home():
    return {"message": "RAG Learning API is running"}


def classify_question(question: str) -> str:
    prompt = f"""
                Classify the user's question into exactly one category:

                - text_only: A normal question that does not need an image.
                - image: The user wants to see an image, screenshot, diagram,
                flowchart, or visual, or asks about what a visual shows.
                - both: The user wants a text explanation and a relevant visual.

                Return only a JSON object with this format:
                {{"intent": "text_only"}}

                Choose "image" when the user mainly asks to see a visual.
                Choose "both" when they request an explanation and a visual.
                Do not choose image just because a visual might be useful.

                User question:
                {question}
        """

    response = openai_client.responses.create(
        model=openai_model,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )

    try:
        intent = json.loads(response.output_text).get("intent", "text_only")
        return intent if intent in {"text_only", "image", "both"} else "text_only"
    except (ValueError, TypeError, AttributeError):
        return "text_only"


def search_images(question: str) -> list[dict]:
    results = image_search_client.search(
        search_text=question,
        query_type="semantic",
        semantic_configuration_name=IMAGE_SEMANTIC_CONFIG,
        top=5,
        select=["image_name", "description", "page_number", "blob_name"],
    )

    images = []
    for result in results:
        score = result.get("@search.reranker_score") or 0.0

        if score < IMAGE_SCORE_THRESHOLD:
            continue

        images.append({
            "name": result["image_name"],
            "description": result["description"],
            "page": result["page_number"],
            "url": f"/api/images/{result['blob_name']}",
            "relevance_score": round(score, 2),
        })

    return images


def build_agent_input(question: str, images: list[dict]) -> str:
    """Adds image descriptions so the agent can answer about the visuals."""
    if not images:
        return question

    image_context = "\n\n".join(
        f"[{image['name']}, page {image['page']}]\n{image['description']}"
        for image in images
    )

    return (
        f"{question}\n\n"
        "The application found these images in the employee handbook "
        "and is displaying them to the user. Use their descriptions "
        "if they help answer the question:\n\n"
        f"{image_context}"
    )


def retrieved_documents(response) -> list[dict]:
    """Chunks returned by the agent's Azure AI Search tool, in order."""
    documents = []

    for item in response.output:
        data = item.model_dump()
        if data.get("type") != "azure_ai_search_call_output":
            continue

        try:
            output = json.loads(data.get("output") or "{}")
            documents.extend(output.get("documents", []))
        except (ValueError, AttributeError):
            print("Could not parse the Azure AI Search tool output.")

    return documents


def lookup_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Gets the page and chunk details for the cited chunk IDs."""
    if not chunk_ids:
        return {}

    # Chunk IDs are hex SHA-256 hashes, so they are safe in a filter.
    results = text_search_client.search(
        search_text="*",
        filter=f"search.in(id, '{','.join(chunk_ids)}', ',')",
        select=["id", "file_name", "page_number", "chunk_number", "content"],
        top=len(chunk_ids),
    )

    return {result["id"]: result for result in results}


def make_snippet(text: str, length: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length].rstrip() + "…"


def add_citations(response) -> tuple[str, list[dict]]:
    """Replaces the agent's 【…†source】 markers with [1], [2], …
    and returns the cited handbook chunks with their page numbers."""
    documents = retrieved_documents(response)
    numbers: dict[str, int] = {}  # chunk id -> citation number
    parts = []

    for item in response.output:
        data = item.model_dump()
        if data.get("type") != "message":
            continue

        for content in data.get("content") or []:
            if content.get("type") != "output_text":
                continue

            text = content.get("text") or ""
            replacements = []

            for annotation in sorted(
                content.get("annotations") or [],
                key=lambda a: a.get("start_index", 0),
            ):
                # The tool titles each citation "doc_N", where N is the
                # position of the chunk in the tool's search results.
                match = re.fullmatch(r"doc_(\d+)", annotation.get("title") or "")
                if annotation.get("type") != "url_citation" or not match:
                    continue

                position = int(match.group(1))
                start = annotation.get("start_index")
                end = annotation.get("end_index")

                if (
                    position >= len(documents)
                    or start is None
                    or end is None
                    or not text[start:end].startswith("【")
                ):
                    continue

                chunk_id = documents[position]["id"]
                numbers.setdefault(chunk_id, len(numbers) + 1)
                replacements.append((start, end, numbers[chunk_id]))

            # Replace from the end so earlier positions stay valid.
            for start, end, number in reversed(replacements):
                text = f"{text[:start]}[{number}]{text[end:]}"

            parts.append(CITATION_MARKER.sub("", text))

    chunks = lookup_chunks(list(numbers))
    citations = []

    for chunk_id, number in numbers.items():
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue

        citations.append({
            "number": number,
            "file_name": chunk["file_name"],
            "page": chunk["page_number"],
            "chunk": chunk["chunk_number"],
            "snippet": make_snippet(chunk["content"]),
            "url": (
                f"/api/documents/{chunk['file_name']}"
                f"#page={chunk['page_number']}"
            ),
        })

    return "".join(parts), citations


@app.post("/ask")
def ask_question(request: QuestionRequest):
    # 1. Decide whether the question needs images.
    try:
        intent = classify_question(request.question)
    except Exception as error:
        print(f"Intent classification failed: {type(error).__name__}: {error}")
        intent = "text_only"

    print(f"Intent: {intent}")

    # 2. Find relevant images in the image index.
    images = []
    if intent in {"image", "both"}:
        try:
            images = search_images(request.question)
        except Exception as error:
            print(f"Image search failed: {type(error).__name__}: {error}")

    # 3. Ask the agent. It searches the text index itself and
    #    receives the image descriptions found above.
    try:
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = agent_client.conversations.create().id

        response = agent_client.responses.create(
            conversation=conversation_id,
            input=build_agent_input(request.question, images),
        )

    except Exception as error:
        print(f"Foundry agent error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=502,
            detail=f"AI agent request failed: {type(error).__name__}: {error}",
        ) from error

    # 4. Turn the agent's source markers into numbered citations.
    try:
        answer, citations = add_citations(response)
    except Exception as error:
        print(f"Citation extraction failed: {type(error).__name__}: {error}")
        answer, citations = CITATION_MARKER.sub("", response.output_text), []

    return {
        "question": request.question,
        "answer": answer,
        "conversation_id": conversation_id,
        "images": images,
        "citations": citations,
    }


def download_blob(name: str) -> tuple[bytes, str]:
    try:
        blob = blob_container.get_blob_client(name)
        properties = blob.get_blob_properties()
        data = blob.download_blob().readall()
    except Exception as error:
        print(f"Blob download failed: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {name}",
        ) from error

    content_type = (
        properties.content_settings.content_type
        or "application/octet-stream"
    )
    return data, content_type


@app.get("/api/images/{image_name}")
def get_image(image_name: str = ApiPath(
    min_length=1,
    max_length=200,
    pattern=r"^[a-zA-Z0-9_.-]+$",
)):
    image_bytes, content_type = download_blob(image_name)

    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="The requested file is not an image.",
        )

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.get("/api/documents/{file_name}")
def get_document(file_name: str = ApiPath(
    min_length=1,
    max_length=200,
    pattern=r"^[a-zA-Z0-9_.-]+$",
)):
    pdf_bytes, content_type = download_blob(file_name)

    if content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="The requested file is not a PDF.",
        )

    # "inline" lets the browser open the PDF, including at #page=N.
    return Response(
        content=pdf_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{file_name}"',
            "Cache-Control": "private, max-age=300",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
