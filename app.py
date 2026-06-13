from utils.pdf_loader import load_pdf
from utils.chunker import chunk_text
from utils.vector_store import create_vector_store
from utils.retriever import retrieve

text = load_pdf("data/sample.pdf")

chunks = chunk_text(text)

index, embeddings = create_vector_store(chunks)

query = input("Ask a question: ")

results = retrieve(query, index, chunks)

print("\nRetrieved Context:\n")

for i, chunk in enumerate(results, 1):
    print(f"Chunk {i}:")
    print(chunk)