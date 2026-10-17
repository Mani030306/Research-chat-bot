# 📚 Research Paper Answer Bot

An end-to-end Retrieval-Augmented Generation (RAG) system built over 8 seminal GenAI and Large Language Model (LLM) research papers (*Attention Is All You Need, BERT, GPT-3, T5, LoRA, RAG, InstructGPT, LLaMA*).

---

## 🏗️ Architecture

```text
PDF Papers (8 arXiv Papers)
  ↓
Document Loader (PyPDFLoader)
  ↓
Text Chunking (RecursiveCharacterTextSplitter, size=1000, overlap=200)
  ↓
Embedding Model (BAAI/bge-small-en-v1.5 & OpenAI text-embedding-3-small experiment)
  ↓
Vector Database (ChromaDB local persistence)
  ↓
Advanced Retrieval (Dense, MMR, Hybrid BM25+Dense, Cross-Encoder Reranker, Multi-Query)
  ↓
Context & Citation Formatting
  ↓
LLM Generation (Google Gemini `gemini-2.5-flash`)
  ↓
Grounded Answer + Top-3 Supporting Citations (Streamlit UI)
```

> **Note on LLM Generation**:
> OpenAI API generation was replaced with Google Gemini API (`langchain-google-genai`) to avoid dependency on exhausted OpenAI API quotas while fully preserving the entire RAG pipeline and retrieval architecture.
>
> **Note on Embeddings**:
> OpenAI `text-embedding-3-small` remains integrated for comparative embedding experiments, while the primary local vector store uses open-source HuggingFace `BAAI/bge-small-en-v1.5` embeddings.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Create a `.env` file in the project root directory (copied from `.env.example`):

```bash
cp .env.example .env
```

Add your Google Gemini API Key:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for comparative embedding experiments)
# OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Ingestion & Vector Indexing
```bash
python main.py
```

### 4. Launch Interactive Streamlit UI
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Key Features

- **Grounded Anti-Hallucination Prompting**: Strictly answers based on retrieved context; falls back to `"I don't know based on the provided research papers."` when insufficient.
- **Top-3 Cited Sources**: Programs each response to display paper title, page number, and source passage excerpt.
- **5 Retrieval Strategies**:
  1. Dense Retrieval (Cosine Similarity)
  2. Max Marginal Relevance (MMR)
  3. Hybrid Search (BM25 + Dense RRF)
  4. Cross-Encoder Neural Reranking (`ms-marco-MiniLM-L-6-v2`)
  5. Multi-Query Retrieval (LLM Query Variants)
- **Sliding-Window Memory**: Retains conversation history (`k=5`) for multi-turn Q&A.
