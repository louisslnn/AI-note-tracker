import chromadb
import uuid
import re
import os
import shutil
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from typing import List
from .save_prompt import save_prompt_as_text
from .get_last_note import get_last_note, get_note

path_to_data_folder = "/Users/louissalanon/Desktop/AI:DS projects/smart-knowledge-tracker/notes_data/"
path_to_vector_store="/Users/louissalanon/Desktop/AI:DS projects/smart-knowledge-tracker/vector_store"

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()

embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=path_to_vector_store)

collection = chroma_client.get_or_create_collection(
    name="database",
    embedding_function=embedding_function
)

def reset_collection():
    # Delete vector store BEFORE loading chroma
    if os.path.exists(path_to_vector_store):
        shutil.rmtree(path_to_vector_store)

    # Recreate the directory to avoid permission issues
    os.makedirs(path_to_vector_store, exist_ok=True)

    # Delete note .txt files
    for filename in os.listdir(path_to_data_folder):
        file_path = os.path.join(path_to_data_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Initialize new client
    chroma_client = chromadb.PersistentClient(path=path_to_vector_store)

    # Create new collection
    return chroma_client.get_or_create_collection(
        name="database",
        embedding_function=embedding_function
    )

def add_note_to_database(path_to_note, last_note=False):
    
    if last_note:
        text = get_last_note()
    else:
        text = get_note(path_to_note)
    unique_id = str(uuid.uuid4())

    collection.add(
        documents=[text],
        metadatas=[{"source": "note"}],
        ids=[unique_id]
    )

    print("Added note to database successfully.")

def query_to_database(query: str = ""):
   
    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=['documents', 'distances']
    )

    return results

def clean_raw_output(raw_output):
    if isinstance(raw_output, list):
        try:
            clean_output = raw_output[0][0].replace('\n', '').strip()
        except (IndexError, TypeError):
            clean_output = str(raw_output).strip()
    else:
        clean_output = str(raw_output).strip()
    
    return clean_output
