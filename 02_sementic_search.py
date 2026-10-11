from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Our document collection
documents = [
    "You can recover your account by resetting your credentials.",
    "The vehicle requires regular oil changes and maintenance.",
    "To regain access, follow the account recovery procedure.",
]

# 3. Generate document embeddings once
document_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)


def search(query, top_k=3):
    # 4. Convert the query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # 5. Calculate cosine similarity
    # Since vectors are normalized, dot product = cosine similarity
    scores = document_embeddings @ query_embedding

    # 6. Sort by highest score
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    # 7. Return matching documents and scores
    return [
        {
            "document": documents[i],
            "score": float(scores[i])
        }
        for i in ranked_indices
    ]


# Try a query with different wording
query = "I can't log in to my account"

for result in search(query):
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['document']}")
    print("-" * 60)
