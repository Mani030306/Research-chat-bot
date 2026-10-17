"""
Embeddings Module — Step 3: Embedding Model Comparison.

Compares two embedding models — one open-source (HuggingFace) and one
commercial (OpenAI) — to determine the best choice for our RAG pipeline.

Models compared:
  1. BAAI/bge-small-en-v1.5 (Open-source, 384 dims, ~33M params)
     - Fast, lightweight, strong MTEB benchmark scores for its size
     - Runs locally — no API costs or latency
  2. OpenAI text-embedding-3-small (Commercial, 1536 dims)
     - State-of-the-art quality, higher dimensional embeddings
     - API-based — costs per token but no local compute needed

Design Decision: We compare embeddings qualitatively by embedding sample
queries and measuring cosine similarity against document chunks. The model
that retrieves more relevant chunks (judged by manual inspection) is chosen
as the default for the final pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import (
    HF_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


# =============================================================================
# 1. Embedding Model Initialization
# =============================================================================

def get_hf_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace (open-source) embedding model.

    Default: BAAI/bge-small-en-v1.5
    - 384 dimensions, ~33M parameters
    - Good balance of quality vs. speed
    - Runs entirely locally

    Args:
        model_name: HuggingFace model identifier. Defaults to config value.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    model_name = model_name or HF_EMBEDDING_MODEL
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use GPU if available: "cuda"
        encode_kwargs={"normalize_embeddings": True},  # L2 normalize for cosine sim
    )
    print(f"[+] HuggingFace embeddings loaded: {model_name}")
    return embeddings


def get_openai_embeddings(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Initialize the OpenAI (commercial) embedding model.

    Default: text-embedding-3-small
    - 1536 dimensions
    - State-of-the-art quality
    - API-based (cost: ~$0.02 per 1M tokens)

    Args:
        model_name: OpenAI model name. Defaults to config value.

    Returns:
        OpenAIEmbeddings instance.
    """
    model_name = model_name or OPENAI_EMBEDDING_MODEL
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured. Please add your key to .env")
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
    )
    print(f"[+] OpenAI embeddings initialized: {model_name}")
    return embeddings


# =============================================================================
# 2. Embedding Comparison
# =============================================================================

# Sample queries spanning different papers for qualitative comparison
COMPARISON_QUERIES = [
    "What is the self-attention mechanism and how does it work?",
    "How does masked language modeling work in BERT?",
    "What is few-shot learning in the context of GPT-3?",
    "How does LoRA reduce the number of trainable parameters?",
    "What is retrieval-augmented generation?",
]


def compute_similarity(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
) -> list[float]:
    """
    Compute cosine similarity between a query and document embeddings.

    Args:
        query_embedding: Query vector.
        doc_embeddings: List of document vectors.

    Returns:
        List of cosine similarity scores.
    """
    query_vec = np.array(query_embedding)
    doc_vecs = np.array(doc_embeddings)

    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

    similarities = doc_norms @ query_norm
    return similarities.tolist()


def compare_embeddings(
    chunks: list[Document],
    queries: Optional[list[str]] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Compare HuggingFace and OpenAI embeddings on sample queries.

    For each query, embeds the query and all chunks using both models,
    retrieves the top-k most similar chunks, and presents them for
    qualitative comparison.

    Args:
        chunks: List of chunked Document objects.
        queries: List of test queries. Defaults to COMPARISON_QUERIES.
        top_k: Number of top chunks to retrieve per query.

    Returns:
        DataFrame with comparison results.
    """
    queries = queries or COMPARISON_QUERIES

    # Initialize both models
    hf_model = get_hf_embeddings()
    openai_model = get_openai_embeddings()

    # Get chunk texts
    chunk_texts = [c.page_content for c in chunks]

    print("\nEmbedding all chunks with HuggingFace model...")
    hf_doc_embeddings = hf_model.embed_documents(chunk_texts)
    print(f"  ✓ {len(hf_doc_embeddings)} chunks embedded (dim={len(hf_doc_embeddings[0])})")

    print("Embedding all chunks with OpenAI model...")
    openai_doc_embeddings = openai_model.embed_documents(chunk_texts)
    print(f"  ✓ {len(openai_doc_embeddings)} chunks embedded (dim={len(openai_doc_embeddings[0])})")

    results = []

    print("\n" + "=" * 80)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 80)

    for query in queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 60)

        # HuggingFace retrieval
        hf_query_emb = hf_model.embed_query(query)
        hf_sims = compute_similarity(hf_query_emb, hf_doc_embeddings)
        hf_top_indices = np.argsort(hf_sims)[-top_k:][::-1]

        print(f"\n  🔷 HuggingFace ({HF_EMBEDDING_MODEL}):")
        for rank, idx in enumerate(hf_top_indices, 1):
            chunk = chunks[idx]
            print(f"    [{rank}] Score: {hf_sims[idx]:.4f} | "
                  f"{chunk.metadata.get('paper_title', 'Unknown')}, "
                  f"Page {chunk.metadata.get('page', '?')}")
            print(f"        \"{chunk.page_content[:120]}...\"")

            results.append({
                "query": query,
                "model": "HuggingFace (BGE)",
                "rank": rank,
                "score": round(hf_sims[idx], 4),
                "paper_title": chunk.metadata.get("paper_title", "Unknown"),
                "page": chunk.metadata.get("page", "?"),
                "chunk_preview": chunk.page_content[:150],
            })

        # OpenAI retrieval
        openai_query_emb = openai_model.embed_query(query)
        openai_sims = compute_similarity(openai_query_emb, openai_doc_embeddings)
        openai_top_indices = np.argsort(openai_sims)[-top_k:][::-1]

        print(f"\n  🟢 OpenAI ({OPENAI_EMBEDDING_MODEL}):")
        for rank, idx in enumerate(openai_top_indices, 1):
            chunk = chunks[idx]
            print(f"    [{rank}] Score: {openai_sims[idx]:.4f} | "
                  f"{chunk.metadata.get('paper_title', 'Unknown')}, "
                  f"Page {chunk.metadata.get('page', '?')}")
            print(f"        \"{chunk.page_content[:120]}...\"")

            results.append({
                "query": query,
                "model": "OpenAI (text-embedding-3-small)",
                "rank": rank,
                "score": round(openai_sims[idx], 4),
                "paper_title": chunk.metadata.get("paper_title", "Unknown"),
                "page": chunk.metadata.get("page", "?"),
                "chunk_preview": chunk.page_content[:150],
            })

    df = pd.DataFrame(results)
    return df


def get_embedding_dimensions_comparison() -> dict:
    """
    Return a comparison table of embedding model properties.
    Useful for documentation in the notebook.
    """
    return {
        "HuggingFace (BGE-small)": {
            "model": HF_EMBEDDING_MODEL,
            "dimensions": 384,
            "parameters": "~33M",
            "cost": "Free (local)",
            "latency": "Fast (CPU/GPU)",
            "type": "Open-source",
        },
        "OpenAI (text-embedding-3-small)": {
            "model": OPENAI_EMBEDDING_MODEL,
            "dimensions": 1536,
            "parameters": "Unknown (proprietary)",
            "cost": "~$0.02/1M tokens",
            "latency": "API call (~100-300ms)",
            "type": "Commercial",
        },
    }
