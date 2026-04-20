import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings # Switched from OpenAI

DATA_PATH = "data"
CHROMA_PATH = "db"

def create_vector_db():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    all_chunks = []

    for filename in files:
        print(f"📄 Loading {filename}...")
        loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"✅ Processing {len(all_chunks)} chunks into the local database...")

    # Using Ollama instead of OpenAI - no API key required!
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db = Chroma.from_documents(
        all_chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"🚀 Success! Database saved to '{CHROMA_PATH}' folder.")

if __name__ == "__main__":
    create_vector_db()