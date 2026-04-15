# 📄 Local PDF RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built entirely with open-source tools. This application allows users to upload PDF documents and ask context-specific questions. All text extraction, embedding, and generation occurs 100% locally on-device, ensuring zero API costs and complete data privacy.

## 🚀 Pipeline Architecture

This project strictly follows the standard RAG architecture:

1. **Data Ingestion:** `PyPDFLoader` extracts text from the uploaded PDF.
2. **Chunking:** `RecursiveCharacterTextSplitter` divides the text into manageable 1000-character chunks with a 200-character overlap to preserve semantic context.
3. **Embedding:** Chunks are vectorized using the local `nomic-embed-text` model via Ollama.
4. **Vector Storage:** Vectors are stored locally in a `Chroma` database for rapid similarity search.
5. **Retrieval & Generation:** User queries are embedded, matched against the vector store using cosine similarity, and passed to a local `Llama 3` model alongside the retrieved context to generate deterministic, grounded answers.

## 🛠️ Tech Stack

* **Language:** Python
* **Orchestration:** LangChain
* **LLM Engine:** Ollama (Llama 3 for generation, Nomic for embeddings)
* **Vector Database:** ChromaDB
* **User Interface:** Gradio

## ⚙️ Local Setup Instructions

### Prerequisites
You must have [Ollama](https://ollama.com/) installed on your machine to run the local models.

Once Ollama is installed, pull the required models via your terminal:
```bash
ollama run llama3
ollama pull nomic-embed-text