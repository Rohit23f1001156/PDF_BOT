import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# We import Ollama modules for 100% local, private, and free execution.
# This demonstrates the ability to build AI pipelines without relying on paid APIs.
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# ==========================================
# CORE RAG LOGIC
# ==========================================

def process_pdf_and_setup_rag(pdf_file_path):
    """
    Ingests a PDF, chunks the text, generates vector embeddings locally, 
    and returns a retrieval chain ready for querying.
    """
    # 1. Extract Text
    # WHY: PyPDFLoader reads the binary PDF file and extracts the raw text pages.
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()

    # 2. Chunking
    # WHY: LLMs have finite context windows. We split the document into smaller chunks (1000 chars)
    # with a 200-char overlap. The overlap ensures context isn't lost if a sentence gets cut in half.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Embeddings and Vector Store
    # WHY: We use 'nomic-embed-text' via Ollama to convert text chunks into mathematical vectors.
    # Chroma DB stores these vectors locally so we can perform fast similarity searches later.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # WHY: The retriever acts as our search engine, returning the top 3 most relevant chunks for any query.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM Setup and Prompt Engineering
    # WHY: We use 'llama3' running locally via Ollama. 
    # Temperature is set to 0 to force factual, deterministic answers (reducing hallucination).
    llm = ChatOllama(model="llama3", temperature=0)

    # WHY: This prompt strictly confines the LLM to only use the retrieved PDF context.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Create Chains
    # WHY: 'Stuff' chain injects all retrieved chunks directly into the {context} variable of our prompt.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # WHY: This links the retriever (data fetcher) to the QA chain (text generator).
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# ==========================================
# UI & STATE MANAGEMENT
# ==========================================

# Global variable to hold our active RAG pipeline in memory
global_rag_chain = None

def upload_file(file):
    """Triggered when a user uploads a PDF."""
    global global_rag_chain
    if file is None:
        return "Please upload a file."
    
    # Build the RAG pipeline with the uploaded file
    global_rag_chain = process_pdf_and_setup_rag(file.name)
    return f"PDF '{os.path.basename(file.name)}' processed successfully! Ask a question."

def answer_question(question):
    """Triggered when a user submits a query."""
    global global_rag_chain
    if global_rag_chain is None:
        return "Please upload a PDF first."
    
    if not question.strip():
        return "Please enter a question."

    # Pass the user's question through the pipeline
    response = global_rag_chain.invoke({"input": question})
    answer = response["answer"]
    
    # Display the specific chunks of text the LLM used to formulate its answer
    sources = "\n\n---\n**Sources Used:**\n"
    for i, doc in enumerate(response["context"]):
        sources += f"{i+1}. {doc.page_content[:100]}...\n"
        
    return answer + sources

# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Local PDF RAG Chatbot (Ollama + LangChain)")
    gr.Markdown("Upload a document and ask questions. 100% private, local, and cost-free execution.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="1. Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            file_input.upload(fn=upload_file, inputs=file_input, outputs=upload_status)
            
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="2. Ask a Question about the PDF")
            submit_btn = gr.Button("Submit Query", variant="primary")
            output_text = gr.Textbox(label="3. Answer & Sources", interactive=False, lines=10)
            
            submit_btn.click(fn=answer_question, inputs=query_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")