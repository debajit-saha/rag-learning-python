"""Image RAG pipeline: extract -> describe -> index -> search."""

import base64
import hashlib
import json
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "employee_handbook.pdf"
IMAGE_DIR = BASE_DIR / "extracted_images"
DESCRIPTION_FILE = BASE_DIR / "image_descriptions.json"

INDEX_NAME = os.environ["AZURE_SEARCH_IMAGE_INDEX"]

# Must match the actual Blob Storage paths of your uploaded images.
BLOB_PREFIX = "rag-documents/"

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def search_credential():
    return AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])


# Step 1: Extract images from the PDF.
def extract_images():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(PDF_PATH)
    image_count = 0

    for page_number, page in enumerate(doc, start=1):
        for image_number, image_info in enumerate(
            page.get_images(full=True), start=1
        ):
            xref = image_info[0]
            image = doc.extract_image(xref)

            extension = image["ext"]
            output_path = IMAGE_DIR / (
                f"page_{page_number}_image_{image_number}.{extension}"
            )

            output_path.write_bytes(image["image"])
            image_count += 1
            print(f"Extracted: {output_path.name}")

    print(f"\nTotal extracted images: {image_count}")
    print(f"Images saved to: {IMAGE_DIR}")


# Step 2: Describe each image with the vision model.
def describe_images():
    client = OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    results = []

    for image_path in sorted(IMAGE_DIR.iterdir()):
        mime_type = MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            continue

        encoded_image = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")

        print(f"Describing {image_path.name}...")

        response = client.responses.create(
            model=deployment,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe images for a document search system. "
                                "Identify the main subject, visible text, "
                                "process steps, labels, and important details. "
                                "Do not invent details that are not visible."
                            ),
                        }
                    ],
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
                            "image_url": (
                                f"data:{mime_type};base64,{encoded_image}"
                            ),
                        },
                    ],
                },
            ],
            max_output_tokens=500,
        )

        description = response.output_text

        results.append({
            "image_name": image_path.name,
            "description": description,
        })

        print(description)
        print("-" * 60)

    DESCRIPTION_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved {len(results)} descriptions to {DESCRIPTION_FILE}")


# Step 3: Create the search index and upload image descriptions.
def index_descriptions():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    credential = search_credential()

    descriptions = json.loads(
        DESCRIPTION_FILE.read_text(encoding="utf-8")
    )

    index_client = SearchIndexClient(
        endpoint=endpoint,
        credential=credential,
    )

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
                name="image-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="image_name"),
                    content_fields=[
                        SemanticField(field_name="description")
                    ],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        semantic_search=semantic_search,
    )

    index_client.create_or_update_index(index)

    print(f"Image index ready with semantic configuration: {INDEX_NAME}")

    documents = []

    for item in descriptions:
        image_name = item["image_name"]

        # Example: page_4_image_1.png -> page 4
        page_number = int(image_name.split("_")[1])

        blob_name = f"{BLOB_PREFIX}{image_name}"

        document_id = hashlib.sha256(
            blob_name.encode("utf-8")
        ).hexdigest()

        documents.append({
            "id": document_id,
            "image_name": image_name,
            "description": item["description"],
            "page_number": page_number,
            "blob_name": blob_name,
        })

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=INDEX_NAME,
        credential=credential,
    )

    results = search_client.upload_documents(documents=documents)

    failed = [r for r in results if not r.succeeded]

    if failed:
        raise RuntimeError(
            f"Failed to index {len(failed)} image documents."
        )

    print(f"Successfully indexed {len(results)} images.")


# Step 4: Interactive search over the image index.
def search_images():
    client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=INDEX_NAME,
        credential=search_credential(),
    )

    while True:
        question = input("\nSearch for an image (or 'exit'): ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        results = client.search(
            search_text=question,
            top=3,
            select=["image_name", "description", "page_number", "blob_name"],
        )

        for result in results:
            print(f"\nImage: {result['image_name']}")
            print(f"Page: {result['page_number']}")
            print(f"Blob: {result['blob_name']}")
            print(f"Description: {result['description']}")


def main():
    extract_images()
    describe_images()
    index_descriptions()
    search_images()


if __name__ == "__main__":
    main()
