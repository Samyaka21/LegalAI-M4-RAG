from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "db"

# 1. Setup the "Prompt" - This tells Llama 3 how to behave
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # 2. Prepare the Database and LLM
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    model = ChatOllama(model="llama3")

    # 3. Ask your legal question
    query_text = "What happens if digital content is not of satisfactory quality?"

    # 4. Retrieval: Find relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Combine the found chunks into one block of text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 5. Generation: Feed context and question to Llama 3
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n🤖 AI is thinking...\n")
    response = model.invoke(prompt)

    # 6. Show the Final Answer
    print("--- LEGAL ASSISTANT ANSWER ---")
    print(response.content)

if __name__ == "__main__":
    main()