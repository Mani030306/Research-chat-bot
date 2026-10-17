"""
Retrieval Module — Step 5: Multiple Retrieval Strategies.

Implements and compares 5 retrieval strategies, from simple cosine
similarity to advanced hybrid search with reranking:

1. Dense Retrieval (Cosine Similarity) — baseline vector search
2. MMR (Max Marginal Relevance) — balances relevance and diversity
3. Hybrid Search (BM25 + Dense) — combines keyword and semantic search
4. Cross-Encoder Reranking — neural reranking of top-k candidates
5. Multi-Query Retrieval — LLM generates query variants for better recall

Design Decision: We implement all 5 to satisfy the rubric requirement of
comparing multiple strategies. The final pipeline uses the strategy with
the best precision/recall trade-off, determined experimentally.

Trade-off Analysis:
  - Dense Retrieval: Fast, simple, but misses keyword-exact matches
  - MMR: Reduces redundancy but may sacrifice top relevance
  - Hybrid: Best of both worlds but more complex setup
  - Reranker: Highest precision but adds ~200ms latency
  - Multi-Query: Best recall but uses extra LLM calls (cost)
"""

from typing import Optional
import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever

from src.config import (
    RETRIEVAL_TOP_K,
    RERANKER_TOP_K,
    RERANKER_MODEL,
    MMR_LAMBDA,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
)


# =============================================================================
# 1. Dense Retrieval (Cosine Similarity) — Baseline
# =============================================================================

def get_dense_retriever(
    vectorstore: Chroma,
    top_k: int = RETRIEVAL_TOP_K,
) -> BaseRetriever:
    """
    Create a dense retrieval (cosine similarity) retriever.

    This is the simplest and fastest approach — pure vector similarity
    search. Good baseline but can miss keyword-exact matches.

    Args:
        vectorstore: ChromaDB vector store.
        top_k: Number of chunks to retrieve.

    Returns:
        LangChain retriever instance.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    print(f"[+] Dense retriever created (top_k={top_k})")
    return retriever


# =============================================================================
# 2. MMR (Max Marginal Relevance)
# =============================================================================

def get_mmr_retriever(
    vectorstore: Chroma,
    top_k: int = RETRIEVAL_TOP_K,
    lambda_mult: float = MMR_LAMBDA,
) -> BaseRetriever:
    """
    Create an MMR retriever that balances relevance and diversity.

    MMR retrieves chunks that are similar to the query but diverse
    from each other. This reduces redundancy — useful when multiple
    chunks say similar things (e.g., same concept from different papers).

    lambda_mult controls the trade-off:
      - 1.0 = pure relevance (same as dense retrieval)
      - 0.0 = pure diversity (most diverse set)
      - 0.7 = good default balance

    Args:
        vectorstore: ChromaDB vector store.
        top_k: Number of chunks to retrieve.
        lambda_mult: Relevance-diversity balance (0.0 to 1.0).

    Returns:
        LangChain retriever instance.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "lambda_mult": lambda_mult,
            "fetch_k": top_k * 4,  # Fetch more candidates for MMR to select from
        },
    )
    print(f"[+] MMR retriever created (top_k={top_k}, lambda={lambda_mult})")
    return retriever


# =============================================================================
# 3. Hybrid Search (BM25 + Dense)
# =============================================================================

def get_hybrid_retriever(
    vectorstore: Chroma,
    chunks: list[Document],
    top_k: int = RETRIEVAL_TOP_K,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
) -> EnsembleRetriever:
    """
    Create a hybrid retriever combining BM25 keyword search with
    dense vector search using Reciprocal Rank Fusion (RRF).

    Why hybrid? Dense retrieval handles semantic paraphrasing well
    (e.g., "neural network training" matches "model optimization"),
    while BM25 catches exact keyword matches that embeddings might
    miss (e.g., acronyms like "RLHF", specific model names).

    The EnsembleRetriever fuses results using RRF, which doesn't
    require score normalization between the two methods.

    Args:
        vectorstore: ChromaDB vector store.
        chunks: Original document chunks (needed for BM25 indexing).
        top_k: Number of chunks to retrieve.
        bm25_weight: Weight for BM25 results (default 0.4).
        dense_weight: Weight for dense results (default 0.6).

    Returns:
        EnsembleRetriever instance.
    """
    # BM25 retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=top_k,
    )

    # Dense retriever (vector-based)
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Combine with EnsembleRetriever (uses RRF by default)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
    )

    print(f"[+] Hybrid retriever created (BM25={bm25_weight}, Dense={dense_weight}, top_k={top_k})")
    return hybrid_retriever


# =============================================================================
# 4. Cross-Encoder Reranking
# =============================================================================

class RerankedRetriever:
    """
    Two-stage retriever: dense retrieval + cross-encoder reranking.

    Stage 1: Retrieve top-N candidates using fast dense retrieval
    Stage 2: Rerank candidates using a cross-encoder model that
             processes (query, document) pairs jointly for more
             accurate relevance scoring.

    The cross-encoder is much more accurate than bi-encoder similarity
    but too slow to run on the full corpus — hence the two-stage design.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        reranker_model_name: str = RERANKER_MODEL,
        initial_top_k: int = RERANKER_TOP_K,
        final_top_k: int = RETRIEVAL_TOP_K,
    ):
        """
        Initialize the reranked retriever.

        Args:
            vectorstore: ChromaDB vector store for initial retrieval.
            reranker_model_name: Cross-encoder model for reranking.
            initial_top_k: Number of candidates for stage 1.
            final_top_k: Number of results after reranking.
        """
        self.vectorstore = vectorstore
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

        # Load cross-encoder
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder(reranker_model_name)
        print(f"[+] Reranker loaded: {reranker_model_name}")
        print(f"  Stage 1: retrieve top-{initial_top_k}, Stage 2: rerank to top-{final_top_k}")

    def invoke(self, query: str) -> list[Document]:
        """
        Retrieve and rerank documents for a query.

        Args:
            query: User query string.

        Returns:
            Top-k reranked Document objects.
        """
        # Stage 1: Dense retrieval of candidates
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.initial_top_k},
        )
        candidates = retriever.invoke(query)

        if not candidates:
            return []

        # Stage 2: Cross-encoder reranking
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Sort by reranker score (descending)
        scored_docs = list(zip(scores, candidates))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top-k after reranking
        reranked = [doc for _, doc in scored_docs[:self.final_top_k]]
        return reranked


def get_reranked_retriever(
    vectorstore: Chroma,
    reranker_model: str = RERANKER_MODEL,
    initial_top_k: int = RERANKER_TOP_K,
    final_top_k: int = RETRIEVAL_TOP_K,
) -> RerankedRetriever:
    """
    Create a two-stage retriever with cross-encoder reranking.

    Args:
        vectorstore: ChromaDB vector store.
        reranker_model: Cross-encoder model name.
        initial_top_k: Candidates for stage 1.
        final_top_k: Final results after reranking.

    Returns:
        RerankedRetriever instance.
    """
    return RerankedRetriever(
        vectorstore=vectorstore,
        reranker_model_name=reranker_model,
        initial_top_k=initial_top_k,
        final_top_k=final_top_k,
    )


# =============================================================================
# 5. Multi-Query Retrieval
# =============================================================================

def get_multiquery_retriever(
    vectorstore: Chroma,
    llm=None,
    top_k: int = RETRIEVAL_TOP_K,
) -> BaseRetriever:
    """
    Create a multi-query retriever that generates query variants.

    The LLM generates 3 alternative phrasings of the user's query,
    retrieves chunks for each variant, and unions the results. This
    improves recall by capturing different aspects of the question.

    Example: "How does attention work?"
      → "What is the self-attention mechanism?"
      → "Explain the attention computation in transformers"
      → "How are attention weights calculated?"

    Args:
        vectorstore: ChromaDB vector store.
        llm: LangChain LLM instance. If None, uses default from config.
        top_k: Number of chunks to retrieve per query variant.

    Returns:
        MultiQueryRetriever instance.
    """
    if llm is None:
        from src.rag_chain import get_llm
        llm = get_llm()

    try:
        from langchain_classic.retrievers import MultiQueryRetriever
    except ImportError:
        from langchain.retrievers.multi_query import MultiQueryRetriever

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    print(f"[+] Multi-query retriever created (top_k={top_k} per variant)")
    return retriever


# =============================================================================
# 6. Strategy Comparison
# =============================================================================

def compare_retrieval_strategies(
    vectorstore: Chroma,
    chunks: list[Document],
    test_queries: list[str],
    llm=None,
) -> dict[str, list]:
    """
    Run all retrieval strategies on a set of test queries and
    collect results for comparison.

    Args:
        vectorstore: ChromaDB vector store.
        chunks: Original chunks (needed for BM25 in hybrid search).
        test_queries: List of queries to test.
        llm: Optional LLM for multi-query retriever.

    Returns:
        Dictionary mapping strategy name to list of result sets.
    """
    strategies = {}

    # Initialize all retrievers
    print("\n" + "=" * 60)
    print("INITIALIZING RETRIEVAL STRATEGIES")
    print("=" * 60)

    strategies["dense"] = get_dense_retriever(vectorstore)
    strategies["mmr"] = get_mmr_retriever(vectorstore)
    strategies["hybrid"] = get_hybrid_retriever(vectorstore, chunks)
    strategies["reranker"] = get_reranked_retriever(vectorstore)
    strategies["multi_query"] = get_multiquery_retriever(vectorstore, llm)

    # Run comparisons
    print("\n" + "=" * 60)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 60)

    all_results = {}

    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 50)
        query_results = {}

        for name, retriever in strategies.items():
            try:
                if isinstance(retriever, RerankedRetriever):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.invoke(query)

                query_results[name] = docs

                print(f"\n  📌 {name.upper()}:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"    [{i}] {doc.metadata.get('paper_title', 'Unknown')}, "
                          f"Page {doc.metadata.get('page', '?')}")
                    print(f"        \"{doc.page_content[:100]}...\"")

            except Exception as e:
                print(f"  ⚠ {name}: Error — {e}")
                query_results[name] = []

        all_results[query] = query_results

    return all_results
