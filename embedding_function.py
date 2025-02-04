from langchain_community.embeddings.ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_embedding_function_llama():
    embeddings = OllamaEmbeddings(model="llama3.2", base_url = os.getenv('HOST'))
    return embeddings