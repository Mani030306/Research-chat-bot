"""
Data Loader Module — Step 1: Data Collection & Document Loading.

Handles downloading seminal GenAI/LLM research papers from arXiv,
loading them with multiple PDF loaders (PyPDFLoader, PDFPlumberLoader),
and inspecting document structure and quality.

Design Decision: We use two different PDF loaders to compare text extraction
quality. PyPDFLoader is the default for its speed and reliability, while
PDFPlumberLoader often extracts tables and formatted text more accurately.
"""

import os
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_core.documents import Document

from src.config import PAPERS_DIR, PAPERS_METADATA


# =============================================================================
# 1. Paper Downloading
# =============================================================================

def download_papers(
    output_dir: Optional[Path] = None,
    force_redownload: bool = False
) -> dict[str, Path]:
    """
    Download all seminal GenAI/LLM research papers from arXiv.

    Args:
        output_dir: Directory to save PDFs. Defaults to PAPERS_DIR from config.
        force_redownload: If True, re-download even if file exists.

    Returns:
        Dictionary mapping paper key to local file path.

    Note:
        ArXiv may rate-limit requests. We add a User-Agent header and
        download sequentially with a small delay to be respectful.
    """
    output_dir = output_dir or PAPERS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    headers = {
        "User-Agent": "ResearchPaperAnswerBot/1.0 (Academic Research Project)"
    }

    for key, meta in tqdm(PAPERS_METADATA.items(), desc="Downloading papers"):
        filename = f"{key}.pdf"
        filepath = output_dir / filename

        if filepath.exists() and not force_redownload:
            print(f"  [+] Already exists: {filename}")
            downloaded[key] = filepath
            continue

        try:
            response = requests.get(meta["url"], headers=headers, timeout=60)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"  [+] Downloaded: {filename} ({len(response.content) / 1024:.1f} KB)")
            downloaded[key] = filepath

        except requests.RequestException as e:
            print(f"  [-] Failed to download {filename}: {e}")

    print(f"\nTotal papers downloaded: {len(downloaded)}/{len(PAPERS_METADATA)}")
    return downloaded


# =============================================================================
# 2. Document Loading with PyPDFLoader
# =============================================================================

def load_papers_pypdf(
    papers_dir: Optional[Path] = None,
) -> list[Document]:
    """
    Load all PDF papers using LangChain's PyPDFLoader.

    Each page becomes a separate Document with metadata:
        - source: original filename
        - page: page number (0-indexed from loader, we convert to 1-indexed)
        - paper_title: human-readable paper title from our metadata

    Args:
        papers_dir: Directory containing PDF files.

    Returns:
        List of LangChain Document objects, one per page.
    """
    papers_dir = papers_dir or PAPERS_DIR
    all_documents = []

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {papers_dir}. "
            "Run download_papers() first."
        )

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs (PyPDFLoader)"):
        paper_key = pdf_path.stem
        paper_meta = PAPERS_METADATA.get(paper_key, {})
        paper_title = paper_meta.get("title", paper_key)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Enrich metadata with paper title and 1-indexed page numbers
        for doc in pages:
            doc.metadata["paper_title"] = paper_title
            doc.metadata["paper_key"] = paper_key
            # PyPDFLoader uses 0-indexed pages; convert to 1-indexed
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            doc.metadata["source"] = pdf_path.name

        all_documents.extend(pages)
        print(f"  [+] {paper_title}: {len(pages)} pages")

    print(f"\nTotal documents loaded: {len(all_documents)} pages across {len(pdf_files)} papers")
    return all_documents


# =============================================================================
# 3. Document Loading with PDFPlumberLoader
# =============================================================================

def load_papers_pdfplumber(
    papers_dir: Optional[Path] = None,
) -> list[Document]:
    """
    Load all PDF papers using LangChain's PDFPlumberLoader.

    PDFPlumberLoader often extracts tables and formatted content more
    accurately than PyPDFLoader. We use this for comparison.

    Args:
        papers_dir: Directory containing PDF files.

    Returns:
        List of LangChain Document objects, one per page.
    """
    papers_dir = papers_dir or PAPERS_DIR
    all_documents = []

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {papers_dir}. "
            "Run download_papers() first."
        )

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs (PDFPlumberLoader)"):
        paper_key = pdf_path.stem
        paper_meta = PAPERS_METADATA.get(paper_key, {})
        paper_title = paper_meta.get("title", paper_key)

        loader = PDFPlumberLoader(str(pdf_path))
        pages = loader.load()

        for doc in pages:
            doc.metadata["paper_title"] = paper_title
            doc.metadata["paper_key"] = paper_key
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            doc.metadata["source"] = pdf_path.name

        all_documents.extend(pages)

    print(f"\nTotal documents loaded (PDFPlumber): {len(all_documents)} pages")
    return all_documents


# =============================================================================
# 4. Document Inspection & Quality Analysis
# =============================================================================

def inspect_documents(documents: list[Document]) -> pd.DataFrame:
    """
    Inspect loaded documents for structure and text quality.

    Generates a summary table with:
        - Paper title
        - Page count
        - Total characters
        - Average characters per page
        - Min/Max characters per page (detects empty/scanned pages)

    Args:
        documents: List of LangChain Document objects.

    Returns:
        pandas DataFrame with inspection results.
    """
    records = []
    papers = {}

    for doc in documents:
        title = doc.metadata.get("paper_title", "Unknown")
        if title not in papers:
            papers[title] = {
                "paper_title": title,
                "pages": 0,
                "total_chars": 0,
                "char_counts": [],
            }
        papers[title]["pages"] += 1
        char_count = len(doc.page_content)
        papers[title]["total_chars"] += char_count
        papers[title]["char_counts"].append(char_count)

    for title, info in papers.items():
        char_counts = info["char_counts"]
        records.append({
            "Paper Title": title,
            "Pages": info["pages"],
            "Total Characters": info["total_chars"],
            "Avg Chars/Page": round(info["total_chars"] / info["pages"]),
            "Min Chars/Page": min(char_counts),
            "Max Chars/Page": max(char_counts),
            "Empty Pages": sum(1 for c in char_counts if c < 50),
        })

    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("DOCUMENT INSPECTION SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Flag potential issues
    total_empty = df["Empty Pages"].sum()
    if total_empty > 0:
        print(f"\n[!] Warning: {total_empty} pages with < 50 characters detected.")
        print("  These may be scanned/image-only pages that need OCR.")

    return df


def get_paper_titles(documents: list[Document]) -> list[str]:
    """Extract unique paper titles from loaded documents."""
    return list(set(doc.metadata.get("paper_title", "Unknown") for doc in documents))
