"""
Configuration module for the Research Paper Answer Bot.

Centralizes all environment variables, constants, file paths, model names,
and random seeds to ensure reproducibility and easy experimentation.
"""

import os
import sys
sys.modules['tensorflow'] = None
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# 1. Load Environment Variables
# =============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Optional: for CRAG stretch goal

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    import warnings
    warnings.warn(
        "GEMINI_API_KEY is not configured in .env file. "
        "Please add your key to .env for Gemini LLM response generation."
    )
    GEMINI_API_KEY = None

if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    OPENAI_API_KEY = None

# =============================================================================
# 2. Random Seeds (Reproducibility)
# =============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# 3. Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. Research Papers Dataset
# =============================================================================
# Seminal GenAI/LLM research papers — arXiv IDs and metadata
PAPERS_METADATA = {
    "attention_is_all_you_need": {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": 2017,
        "url": "https://arxiv.org/pdf/1706.03762",
    },
    "bert": {
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Devlin et al.",
        "year": 2018,
        "url": "https://arxiv.org/pdf/1810.04805",
    },
    "gpt3": {
        "arxiv_id": "2005.14165",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown et al.",
        "year": 2020,
        "url": "https://arxiv.org/pdf/2005.14165",
    },
    "t5": {
        "arxiv_id": "1910.10683",
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
        "authors": "Raffel et al.",
        "year": 2019,
        "url": "https://arxiv.org/pdf/1910.10683",
    },
    "lora": {
        "arxiv_id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al.",
        "year": 2021,
        "url": "https://arxiv.org/pdf/2106.09685",
    },
    "rag_paper": {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al.",
        "year": 2020,
        "url": "https://arxiv.org/pdf/2005.11401",
    },
    "instructgpt": {
        "arxiv_id": "2203.02155",
        "title": "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)",
        "authors": "Ouyang et al.",
        "year": 2022,
        "url": "https://arxiv.org/pdf/2203.02155",
    },
    "llama": {
        "arxiv_id": "2302.13971",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Touvron et al.",
        "year": 2023,
        "url": "https://arxiv.org/pdf/2302.13971",
    },
}

# =============================================================================
# 5. Chunking Parameters
# =============================================================================
# These are the default parameters; experiments will vary these.
CHUNK_SIZE = 1000           # characters (~250 tokens)
CHUNK_OVERLAP = 200         # characters of overlap between chunks
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # for RecursiveCharacterTextSplitter

# =============================================================================
# 6. Embedding Model Names
# =============================================================================
# Open-source model (HuggingFace) — lightweight, good MTEB scores
HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Commercial model (OpenAI) — state-of-the-art
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# =============================================================================
# 7. LLM Configuration (Google Gemini)
# =============================================================================
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0      # Deterministic for reproducibility
LLM_MAX_TOKENS = 1500       # Sufficient for detailed answers with citations

# =============================================================================
# 8. Retrieval Parameters
# =============================================================================
RETRIEVAL_TOP_K = 5         # Number of chunks to retrieve
RERANKER_TOP_K = 20         # Candidates to retrieve before reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MMR_LAMBDA = 0.7            # Balance between relevance and diversity
HYBRID_BM25_WEIGHT = 0.4    # BM25 weight in hybrid search
HYBRID_DENSE_WEIGHT = 0.6   # Dense retrieval weight in hybrid search

# =============================================================================
# 9. ChromaDB Configuration
# =============================================================================
CHROMA_COLLECTION_HF = "papers_bge_small"
CHROMA_COLLECTION_OPENAI = "papers_openai_small"

# =============================================================================
# 10. Memory Configuration (Stretch Goal 1)
# =============================================================================
CONVERSATION_WINDOW_SIZE = 5  # Number of past exchanges to retain
