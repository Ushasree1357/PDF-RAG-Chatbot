from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve(query, index, chunks, top_k=3):

    query_embedding = model.encode([query])

    k = min(top_k, len(chunks))

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results