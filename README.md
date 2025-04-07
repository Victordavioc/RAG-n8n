## Retrieval-Augmented Generation (RAG) Projects

This repository contains two different approaches for implementing Retrieval-Augmented Generation (RAG) systems:

1. **RAG using n8n (No-code/Low-code platform)**
2. **RAG using LangChain (Code-based implementation)**

---

## 📁 Projects Overview

### 1. RAG with n8n

This project demonstrates how to build a simple RAG pipeline using [n8n](https://n8n.io/), a workflow automation tool.

**Main Features:**
- Upload and extract text from PDF documents
- Split text into chunks for better context
- Generate vector embeddings using OpenAI
- Store and retrieve data from a vector database
- Use a chatbot to answer user queries based on retrieved context

**Technologies Used:**
- n8n
- OpenAI API
- Vector Database (e.g., Pinecone, Weaviate, or a local one like ChromaDB)
- HTTP/Webhook nodes

### 2. RAG with LangChain

This project provides a programmatic RAG implementation using Python and [LangChain](https://www.langchain.com/).

**Main Features:**
- PDF text ingestion
- Text chunking and embedding
- Vector store integration
- Retrieval of relevant chunks at query time
- Generation of answers using OpenAI LLMs

**Technologies Used:**
- Python
- LangChain
- OpenAI
- FAISS or Chroma for vector storage
- Streamlit or CLI for interaction (optional)

---

## 🛠️ How to Run

### RAG with n8n
1. Clone the repository
2. Set up your local n8n instance or use the cloud version
3. Import the provided workflow file
4. Configure API keys and credentials (OpenAI, vector DB, etc.)
5. Start the workflow and interact via the webhook or UI nodes

### RAG with LangChain
1. Clone the repository
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
