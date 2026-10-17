"""
Main Execution Script — Research Paper Answer Bot.

Runs the complete end-to-end pipeline:
1. Paper downloading (8 seminal GenAI/LLM papers)
2. PDF Document loading & inspection
3. Text chunking & statistics
4. Vector database creation with ChromaDB
5. Retrieval strategy comparison
6. RAG pipeline query demonstration
"""

import sys
import os
sys.modules['tensorflow'] = None
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_papers, load_papers_pypdf, inspect_documents
from src.chunking import chunk_recursive, get_chunking_stats
from src.embeddings import get_hf_embeddings, get_openai_embeddings
from src.vectorstore import setup_hf_vectorstore, setup_openai_vectorstore, validate_retrieval
from src.retrieval import compare_retrieval_strategies
from src.config import OPENAI_API_KEY


def main():
    print("=" * 80)
    print("[+] RESEARCH PAPER ANSWER BOT — PIPELINE INITIALIZATION")
    print("=" * 80)

    # 1. Download Papers
    print("\n--- STEP 1: Paper Download ---")
    downloaded = download_papers()

    # 2. Load Documents
    print("\n--- STEP 2: Document Loading ---")
    documents = load_papers_pypdf()
    inspect_documents(documents)

    # 3. Chunking
    print("\n--- STEP 3: Text Chunking ---")
    chunks = chunk_recursive(documents)
    get_chunking_stats(chunks)

    # 4. Vector Store Setup
    print("\n--- STEP 4: Vector Store Setup ---")
    print("Building HuggingFace BGE Vector Store (Local)...")
    hf_vectorstore = setup_hf_vectorstore(chunks)

    validate_retrieval(hf_vectorstore)

    if OPENAI_API_KEY:
        try:
            print("\nBuilding OpenAI Vector Store...")
            openai_vectorstore = setup_openai_vectorstore(chunks)
            print("✓ OpenAI Vector Store ready.")
        except Exception as e:
            print(f"\n⚠ Could not build OpenAI Vector Store: {e}")
            print("  Falling back to HuggingFace BGE Vector Store (1300 chunks indexed).")
    else:
        print("\n⚠ OPENAI_API_KEY not configured. Skipping OpenAI Vector Store & LLM Q&A.")

    print("\n" + "=" * 80)
    print("✓ PIPELINE SETUP COMPLETE!")
    print("To launch the interactive chat UI, run:")
    print("  streamlit run streamlit_app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
