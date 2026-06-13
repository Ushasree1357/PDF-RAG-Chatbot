import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

def get_answer(context, question):

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer based only on the context.
    """

    response = model.generate_content(prompt)

    return response.text