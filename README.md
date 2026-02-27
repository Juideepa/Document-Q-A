# 🤖 DocMind AI – Intelligent Document Q&A Assistant

DocMind AI is a Retrieval-Augmented Generation (RAG) based Generative AI application that allows users to upload PDF documents and interact with them conversationally.
The system retrieves relevant document content using semantic search and generates accurate, context-aware responses using a Large Language Model (LLM).

---

## 🚀 Live Application

🔗 **Live Demo:**  
https://juideepa-docmind-ai.streamlit.app/  

> ⚠️ Note: Since the app is deployed on Streamlit Cloud (free tier), it may go to sleep after inactivity. If prompted, click **“Yes, get this app back up!”** to restart the application.

---

## 🏗️ System Architecture (RAG Pipeline)

1. User uploads PDF document(s)
2. Documents are split into chunks
3. Embeddings generated using Google Gemini Embedding Model
4. Stored in FAISS Vector Database
5. User submits question
6. Relevant chunks retrieved via similarity search
7. Context passed to Groq-hosted LLM
8. LLM generates grounded response

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI & Cloud Deployment
- **LangChain** – LLM Orchestration
- **Groq API** – LLM inference
- **Google Generative AI API** – Embeddings
- **FAISS** – Vector database
- **PyPDFLoader** – Document parsing

---

## 🔑 Environment Variables

For local development, create a `.env` file:
