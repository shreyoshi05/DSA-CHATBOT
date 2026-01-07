# DSA-GPT 🚀  
**Your AI Study Buddy for Cracking Coding Interviews**

DSA-GPT is a Retrieval-Augmented Generation (RAG) based application designed to help students prepare for coding interviews using curated Data Structures & Algorithms (DSA) resources.

The system ingests online articles, indexes them using vector embeddings, and answers user queries with context-aware, interview-focused explanations.

---

## ✨ Features

- 🔗 Ingests DSA articles via URLs  
- 📄 Splits and indexes content using vector embeddings  
- 🔍 Semantic search using FAISS vector database  
- 🤖 Context-aware question answering (RAG)  
- 📌 Displays source links for transparency  
- 🧠 Interview-oriented explanations  

---

## 🏗️ System Architecture (RAG Pipeline)

```text
User Question
      ↓
Vector Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Context Retrieval
      ↓
LLM-based Answer Generation

##🛠️ Tech Stack

Frontend: Streamlit

Backend: Python

LLM: OpenAI (API-based, configurable)

Vector Database: FAISS

Framework: LangChain

Embeddings: OpenAI Embeddings

##📁 Project Structure
DSA-GPT/
├── app.py
├── config/
│   └── settings.py
├── loaders/
│   └── url_loader.py
├── embeddings/
│   └── embedder.py
├── chains/
│   └── qa_chain.py
├── requirements.txt
├── .gitignore
└── README.md


