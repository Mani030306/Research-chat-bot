"""
Vector Store Module — Step 4: Vector Database Setup.

Sets up ChromaDB as the vector database for indexing document chunks
with their embeddings and metadata.

Design Decision: ChromaDB was chosen because:
  - Local-first: no infrastructure, no external services needed
  - Persistent: vector store saves to disk, no rebuild on restart
  - Native LangChain integration via langchain-chroma
  - Metadata filtering support (filter by paper, page, etc.)
  - Production-viable: can scale to millions of vectors

We create separate collections for each embedding model to enable
fair comparison without re-embedding.
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_HF,
    CHROMA_COLLECTION_OPENAI,
    RETRIEVAL_TOP_K,
)


# =============================================================================
# 1. Create Vector Store from Chunks
# =============================================================================

def create_vectorstore(
    chunks: list[Document],
    embedding_model,
    collection_name: str,
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks.

    Indexes all chunks with their embeddings and metadata (source,
    page, paper_title, chunk_id). The store is automatically persisted
    to disk so it doesn't need to be rebuilt each run.

    Args:
        chunks: List of chunked Document objects with metadata.
        embedding_model: LangChain embedding model instance.
        collection_name: Name for the ChromaDB collection.
        persist_directory: Directory for persistent storage.

    Returns:
        ChromaDB vector store instance.
    """
    persist_dir = str(persist_directory or CHROMA_DB_DIR)

    print(f"Creating ChromaDB collection: '{collection_name}'")
    print(f"  Indexing {len(chunks)} chunks...")
    print(f"  Persist directory: {persist_dir}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    print(f"  [+] Successfully indexed {len(chunks)} chunks")
    print(f"  [+] Collection '{collection_name}' persisted to disk")

    return vectorstore


# =============================================================================
# 2. Load Existing Vector Store
# =============================================================================

def load_vectorstore(
    embedding_model,
    collection_name: str,
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """
    Load a previously persisted ChromaDB vector store.

    This allows the pipeline to skip re-embedding on subsequent runs,
    significantly reducing startup time and API costs.

    Args:
        embedding_model: LangChain embedding model instance (must match
            the model used during creation).
        collection_name: Name of the ChromaDB collection to load.
        persist_directory: Directory where the store is persisted.

    Returns:
        ChromaDB vector store instance.
    """
    persist_dir = str(persist_directory or CHROMA_DB_DIR)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

    # Verify the collection has data
    count = vectorstore._collection.count()
    print(f"[+] Loaded collection '{collection_name}': {count} vectors")

    return vectorstore


# =============================================================================
# 3. Vector Store Setup (Convenience Functions)
# =============================================================================

def setup_hf_vectorstore(
    chunks: list[Document],
    force_rebuild: bool = False,
) -> Chroma:
    """
    Set up ChromaDB with HuggingFace BGE embeddings.

    Args:
        chunks: Document chunks to index.
        force_rebuild: If True, rebuild even if store exists.

    Returns:
        ChromaDB vector store with HuggingFace embeddings.
    """
    from src.embeddings import get_hf_embeddings

    embedding_model = get_hf_embeddings()
    persist_dir = CHROMA_DB_DIR

    # Check if collection already exists
    if not force_rebuild:
        try:
            vs = load_vectorstore(embedding_model, CHROMA_COLLECTION_HF, persist_dir)
            if vs._collection.count() > 0:
                print(f"Using existing HuggingFace vector store ({vs._collection.count()} vectors)")
                return vs
        except Exception:
            pass

    return create_vectorstore(chunks, embedding_model, CHROMA_COLLECTION_HF, persist_dir)


def setup_openai_vectorstore(
    chunks: list[Document],
    force_rebuild: bool = False,
) -> Chroma:
    """
    Set up ChromaDB with OpenAI embeddings.

    Args:
        chunks: Document chunks to index.
        force_rebuild: If True, rebuild even if store exists.

    Returns:
        ChromaDB vector store with OpenAI embeddings.
    """
    from src.embeddings import get_openai_embeddings

    embedding_model = get_openai_embeddings()
    persist_dir = CHROMA_DB_DIR

    if not force_rebuild:
        try:
            vs = load_vectorstore(embedding_model, CHROMA_COLLECTION_OPENAI, persist_dir)
            if vs._collection.count() > 0:
                print(f"Using existing OpenAI vector store ({vs._collection.count()} vectors)")
                return vs
        except Exception:
            pass

    return create_vectorstore(chunks, embedding_model, CHROMA_COLLECTION_OPENAI, persist_dir)


# =============================================================================
# 4. Retrieval Validation
# =============================================================================

def validate_retrieval(
    vectorstore: Chroma,
    test_queries: Optional[list[str]] = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> None:
    """
    Validate that the vector store returns semantically relevant chunks.

    Runs a set of test queries and prints the top-k results for
    manual inspection.

    Args:
        vectorstore: ChromaDB vector store instance.
        test_queries: List of test queries. Defaults to basic sanity checks.
        top_k: Number of results to retrieve per query.
    """
    if test_queries is None:
        test_queries = [
            "What is the transformer architecture?",
            "How does attention work in neural networks?",
            "What is BERT and how is it pre-trained?",
        ]

    print("\n" + "=" * 60)
    print("RETRIEVAL VALIDATION")
    print("=" * 60)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 50)

        results = retriever.invoke(query)
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('paper_title', 'Unknown')}, "
                  f"Page {doc.metadata.get('page', '?')}")
            print(f"      \"{doc.page_content[:120]}...\"")

    print("\n[+] Retrieval validation complete. Review results above for relevance.")
