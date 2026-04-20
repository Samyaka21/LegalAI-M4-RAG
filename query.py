from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "db"

def query_database():
    # 1. Prepare the same embedding tool we used in ingest.py
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Ask a legal question
    query_text = "What are the consumer rights regarding digital content?"
    
    # 4. Search the database
    print(f"\n🔍 Searching for: '{query_text}'...")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # 5. Show the results
    print("\n--- TOP RELEVANT LEGAL CLAUSES ---\n")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} (Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:300]}...") # Shows first 300 characters
        print("-" * 30)

if __name__ == "__main__":
    query_database()