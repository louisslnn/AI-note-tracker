import os
import shutil
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Embedding wrapper ---
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

embedding_function = SentenceTransformerEmbeddingFunction(MODEL_NAME)


# --- (1) Initialize Chroma vector store ---
def initialize_vector_store():
    return Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_function)


# --- (2) Reset Chroma DB (delete + re-init) ---
def reset_vector_store():
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    return initialize_vector_store()


# --- (3) Add a note ---
def add_note_to_vector_store(note_text: str, source_name: str):
    vector_store = initialize_vector_store()
    doc = Document(page_content=note_text, metadata={"source": source_name})
    vector_store.add_documents([doc])
    vector_store.persist()


# --- (4) Retrieve with RAG chatbot logic ---
def retrieve_answer_from_notes(query: str) -> str:
    vector_store = initialize_vector_store()
    results = vector_store.similarity_search(query, k=5)

    if not results:
        return "I couldn't find any relevant notes to answer that question."

    notes = [doc.page_content for doc in results]

    template = (
        "You are an AI assistant helping someone recall and reflect on their personal notes.\n\n"
        "When answering a question, always begin with a friendly introduction like:\n"
        "\"Sure! Here's what I found from your notes.\"\n\n"
        "Then include the most relevant notes.\n"
        "After that, offer thoughtful suggestions or a reflection based on them.\n\n"
        "Relevant notes:\n{notes}\n\n"
        "Question: {question}\n\n"
        "Your full structured answer:"
    )

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.2")
    chain = prompt | model

    return chain.invoke({"notes": notes, "question": query})