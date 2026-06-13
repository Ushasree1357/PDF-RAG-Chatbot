import streamlit as st

from utils.pdf_loader import load_pdf
from utils.chunker import chunk_text
from utils.vector_store import create_vector_store
from utils.retriever import retrieve
from utils.gemini_helper import get_answer

st.title("PDF RAG Chatbot")

# Load PDF
text = load_pdf("data/sample.pdf")

# Chunk text
chunks = chunk_text(text)

# Create vector store
index, embeddings = create_vector_store(chunks)

# User Question
question = st.text_input("Ask a Question")

if question:
    results = retrieve(
        question,
        index,
        chunks
    )

    context = "\n".join(results)

    st.subheader("Retrieved Context")
    st.write(context)

    st.write("Calling Gemini...")

    answer = get_answer(context, question)

    st.subheader("Answer")
    st.write(answer)