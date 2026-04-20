import gradio as gr
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Setup Brain ---
CHROMA_PATH = "db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
model = ChatOllama(model="llama3")

# --- 2. Ingestion Logic ---
def add_new_pdf(file_obj):
    if file_obj is None: return "No file uploaded."
    try:
        loader = PyPDFLoader(file_obj.name)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        db.add_documents(chunks)
        return f"✅ Successfully added {len(chunks)} chunks to the brain!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- 3. Day 14: Memory-Aware Chat Logic ---
def legal_chat(message, history):
    # Convert Gradio history into LangChain objects
    chat_history = []
    for turn in history:
        if turn["role"] == "user":
            chat_history.append(HumanMessage(content=turn["content"]))
        else:
            chat_history.append(AIMessage(content=turn["content"]))

    # Step A: Rephrase query to be standalone if history exists
    if chat_history:
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the conversation above, rephrase the follow-up question to be a standalone question. If it is already standalone, just return the original question.")
        ])
        rephrased_query = model.invoke(rephrase_prompt.format(chat_history=chat_history, input=message)).content
    else:
        rephrased_query = message

    # Step B: Search the database with the rephrased query
    results = db.similarity_search_with_relevance_scores(rephrased_query, k=10)
    valid_results = [res for res in results if res[1] > 0.3]
    
    if not valid_results:
        response_content = "I couldn't find any relevant legal documents for that specific query."
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in valid_results])
        sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc, _ in valid_results]))
        
        # Step C: Final Answer Generation with History
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional UK Legal Assistant. Use the provided context to answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context from documents:\n{context}\n\nQuestion: {input}")
        ])
        
        response = model.invoke(final_prompt.format(
            chat_history=chat_history, 
            context=context_text, 
            input=message
        ))
        response_content = f"{response.content}\n\n**📚 Sources:** {', '.join(sources)}"
    
    return response_content

# --- 4. Custom UI Design ---
with gr.Blocks() as demo:
    gr.Markdown("# ⚖️ Collaborative Legal AI (Conversational Edition)")
    
    with gr.Row():
        with gr.Column(scale=3):
            # REMOVED type="messages" because Gradio 6 already defaults to it
            gr.ChatInterface(fn=legal_chat)
            
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Add Documents")
            file_upload = gr.File(label="Upload Legal PDF")
            upload_button = gr.Button("Process & Add to Brain")
            status = gr.Textbox(label="Status", interactive=False)

    upload_button.click(add_new_pdf, inputs=[file_upload], outputs=[status])

if __name__ == "__main__":
    demo.launch(share=True, theme="soft")