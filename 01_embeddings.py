from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I want to buy a car.",
    "I want to purchase a vehicle.",
    "How do I cook rice?"
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])

print("Number of dimensions:", len(embeddings[0]))


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


print("\nSimilarity scores:")

print(
    "Car vs vehicle:",
    round(cosine_similarity(embeddings[0], embeddings[1]), 4)
)

print(
    "Car vs cooking:",
    round(cosine_similarity(embeddings[0], embeddings[2]), 4)
)
