
# RAG Learning Project

A hands-on learning project exploring embeddings, semantic search,
document processing, Azure AI Search, and Microsoft Foundry agents
using Python.

## Learning Modules

1. Embeddings fundamentals
2. Semantic search
3. Document processing and chunking
4. Local document search
5. Azure Blob Storage and Azure AI Search
6. Chat with a Microsoft Foundry agent

## Architecture

PDF documents -> Text extraction -> Chunking -> Embeddings -> Azure AI Search -> Foundry agent -> Answer

## Technologies

- Python
- NumPy
- Sentence Transformers
- PyPDF
- Azure Blob Storage
- Azure AI Search
- Microsoft Foundry
- Azure Identity

## Setup

1. Clone this repository.
2. Create a virtual environment.
3. Install the dependencies.
4. Copy `.env.example` to `.env`.
5. Configure your Azure resource settings.
6. Authenticate with Azure if required.

## Security

Never commit `.env`, API keys, storage connection strings,
or other credentials.
