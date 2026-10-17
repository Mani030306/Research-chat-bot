"""
Chunking Module — Step 2: Text Chunking Strategies.

Implements and compares three chunking approaches for splitting research
paper documents into retrieval-ready chunks:

1. Fixed-Size Chunking (CharacterTextSplitter)
2. Recursive Character Text Splitting (RecursiveCharacterTextSplitter)
3. Semantic Chunking (SemanticChunker — splits on topic boundaries)

Design Decision: chunk_size=1000 chars (~250 tokens) with 200-char overlap
is chosen as the default because academic papers have dense, technical content.
Smaller chunks lose context (e.g., a theorem without its proof), while larger
chunks dilute retrieval precision. The 20% overlap ensures continuity at
chunk boundaries.

Critical Requirement: Every chunk retains full metadata (source, page,
paper_title, chunk_id) — this is mandatory for citation generation.
"""

from typing import Optional
import pandas as pd

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


# =============================================================================
# 1. Fixed-Size Chunking
# =============================================================================

def chunk_fixed_size(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents using fixed-size character chunking.

    This is the simplest approach — splits text at exactly `chunk_size`
    characters regardless of content structure. Can break mid-sentence
    or mid-word, which is why we use overlap.

    Args:
        documents: List of page-level Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = CharacterTextSplitter(
        separator=" ",          # Split on spaces (avoids mid-word breaks)
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk IDs for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"fixed_{i}"
        chunk.metadata["chunking_strategy"] = "fixed_size"

    print(f"Fixed-size chunking: {len(documents)} pages → {len(chunks)} chunks")
    return chunks


# =============================================================================
# 2. Recursive Character Text Splitting
# =============================================================================

def chunk_recursive(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Optional[list[str]] = None,
) -> list[Document]:
    """
    Split documents using recursive character text splitting.

    This is the recommended approach for most RAG pipelines. It tries to
    split on paragraph breaks first, then sentences, then words — preserving
    the most semantic coherence possible within each chunk.

    Separator hierarchy: ["\\n\\n", "\\n", ". ", " ", ""]
    - First tries double newlines (paragraph boundaries)
    - Falls back to single newlines, then sentences, then words

    Args:
        documents: List of page-level Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        separators: Custom separator hierarchy (defaults to config).

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or CHUNK_SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"recursive_{i}"
        chunk.metadata["chunking_strategy"] = "recursive"

    print(f"Recursive chunking: {len(documents)} pages → {len(chunks)} chunks")
    return chunks


# =============================================================================
# 3. Semantic Chunking
# =============================================================================

def chunk_semantic(
    documents: list[Document],
    embedding_model=None,
) -> list[Document]:
    """
    Split documents using semantic chunking (topic-boundary detection).

    This approach uses embeddings to detect topic shifts within the text,
    splitting at natural semantic boundaries rather than fixed character
    counts. This produces variable-length chunks that align with the
    document's actual topic structure.

    Requires an embedding model to compute sentence similarities.

    Args:
        documents: List of page-level Document objects.
        embedding_model: LangChain embedding model instance.
            If None, uses HuggingFace BGE embeddings.

    Returns:
        List of chunked Document objects with preserved metadata.

    Note:
        Semantic chunking is more expensive (requires embedding computation)
        and may produce very large or very small chunks. Monitor the
        chunk size distribution carefully.
    """
    if embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.config import HF_EMBEDDING_MODEL
        embedding_model = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    try:
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,  # Split at top-25% similarity drops
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"semantic_{i}"
            chunk.metadata["chunking_strategy"] = "semantic"

        print(f"Semantic chunking: {len(documents)} pages → {len(chunks)} chunks")
        return chunks

    except ImportError:
        print("⚠ langchain-experimental not installed. Falling back to recursive chunking.")
        return chunk_recursive(documents)


# =============================================================================
# 4. Chunking Strategy Comparison
# =============================================================================

def compare_chunking_strategies(
    documents: list[Document],
    embedding_model=None,
) -> dict[str, list[Document]]:
    """
    Run all three chunking strategies and return results for comparison.

    Args:
        documents: List of page-level Document objects.
        embedding_model: Embedding model for semantic chunking.

    Returns:
        Dictionary mapping strategy name to list of chunks.
    """
    results = {}

    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    results["fixed_size"] = chunk_fixed_size(documents)
    results["recursive"] = chunk_recursive(documents)
    results["semantic"] = chunk_semantic(documents, embedding_model)

    return results


def get_chunking_stats(chunks: list[Document]) -> pd.DataFrame:
    """
    Generate statistics for a set of chunks.

    Returns a DataFrame with per-paper chunk counts and character distributions.
    """
    records = []
    for chunk in chunks:
        records.append({
            "paper_title": chunk.metadata.get("paper_title", "Unknown"),
            "page": chunk.metadata.get("page", 0),
            "chunk_length": len(chunk.page_content),
            "strategy": chunk.metadata.get("chunking_strategy", "unknown"),
        })

    df = pd.DataFrame(records)

    # Summary statistics
    summary = df.groupby("strategy").agg(
        total_chunks=("chunk_length", "count"),
        avg_length=("chunk_length", "mean"),
        min_length=("chunk_length", "min"),
        max_length=("chunk_length", "max"),
        std_length=("chunk_length", "std"),
    ).round(0)

    print("\nChunking Statistics:")
    print(summary.to_string())

    return df
