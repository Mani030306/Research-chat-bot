"""
RAG Chain Module — Step 6: RAG Pipeline Construction.

Builds the complete RAG pipeline using LangChain's LCEL (LangChain
Expression Language). The pipeline:

1. Accepts a user question
2. Retrieves relevant chunks from the vector store
3. Formats chunks with citation metadata (paper title, page number)
4. Generates a grounded answer using the LLM
5. Returns the answer with top-3 supporting citations

Prompt Engineering Requirements (from rubric):
  - Answer ONLY from the provided context (no hallucination)
  - Say "I don't know" if context is insufficient
  - Include source paper title, page number, and passage excerpts
  - Support multi-turn conversation (via chat_history placeholder)
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from src.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    GEMINI_API_KEY,
    RETRIEVAL_TOP_K,
)


# =============================================================================
# 1. LLM Initialization
# =============================================================================

def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """
    Initialize the Gemini LLM for response generation.

    Default: gemini-2.5-flash — cost-effective with strong instruction
    following, ideal for RAG where we want faithful, grounded answers.

    Args:
        model: Model name. Defaults to config value (gemini-2.5-flash).
        temperature: Sampling temperature. 0.0 for deterministic output.
        max_tokens: Maximum output response tokens.

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please create a .env file with your GEMINI_API_KEY. "
            "See .env.example for the template."
        )
    return ChatGoogleGenerativeAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_output_tokens=max_tokens or LLM_MAX_TOKENS,
        google_api_key=GEMINI_API_KEY,
    )


# =============================================================================
# 2. Document Formatting for Citations
# =============================================================================

def format_docs_with_citations(docs: list[Document]) -> str:
    """
    Format retrieved documents with citation metadata for the LLM prompt.

    Each chunk is formatted with its source paper title, page number,
    and a unique reference tag that the LLM can use for citations.

    Args:
        docs: List of retrieved Document objects.

    Returns:
        Formatted string with all chunks and their metadata.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        paper_title = doc.metadata.get("paper_title", "Unknown Paper")
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")

        formatted.append(
            f"[Source {i}]\n"
            f"Paper: {paper_title}\n"
            f"Page: {page}\n"
            f"File: {source}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n---\n".join(formatted)


def extract_source_info(docs: list[Document], top_k: int = 3) -> list[dict]:
    """
    Extract source citation information from retrieved documents.

    Returns structured metadata for the top-k sources used in
    generating the response.
    """
    sources = []
    for i, doc in enumerate(docs[:top_k], 1):
        score = doc.metadata.get("relevance_score", doc.metadata.get("score", None))
        sources.append({
            "rank": i,
            "paper_title": doc.metadata.get("paper_title", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "source_file": doc.metadata.get("source", "unknown"),
            "passage": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "full_passage": doc.page_content,
            "score": round(float(score), 3) if score is not None else None,
        })
    return sources


# =============================================================================
# 3. Prompt Template & Answer Modes
# =============================================================================

ANSWER_MODE_INSTRUCTIONS = {
    "standard": "Provide a detailed, technical answer appropriate for a research audience.",
    "concise": "Provide a brief, direct, and bulleted summary (under 150 words).",
    "detailed": "Provide an exhaustive, in-depth academic explanation covering mechanisms, mathematical intuition, and nuances.",
    "viva": "Structure your answer specifically for a technical viva exam:\n1. Core Definition\n2. How It Works (Step-by-Step)\n3. Key Advantages & Benchmark Strengths\n4. Known Limitations / Trade-offs\n5. Viva Takeaway & Related RAG Concepts.",
}


# Main RAG prompt — carefully engineered for grounded, cited responses
RAG_PROMPT_TEMPLATE = """You are a knowledgeable research paper expert assistant. Your task is to answer questions about AI and machine learning research papers using ONLY the provided context.

STRICT RULES:
1. Answer ONLY based on the information in the provided context below. Do not use outside knowledge.
2. If the context does not contain sufficient information to answer the question, respond with: "I don't know based on the provided research papers."
3. Do NOT make up or hallucinate any information, paper titles, citations, or page numbers not present in the context.
4. Always cite your sources by referencing the paper title and page number.
5. Be detailed and technical in your responses — this is for a research audience.

CONTEXT (Retrieved from research papers):
{context}

QUESTION: {question}

Provide a comprehensive answer with citations. After your answer, list the top sources used in this exact format:

**Sources Used:**
1. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
2. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
3. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
"""

# Conversational RAG prompt (includes chat history)
CONVERSATIONAL_RAG_PROMPT_TEMPLATE = """You are a knowledgeable research paper expert assistant. Your task is to answer questions about AI and machine learning research papers using ONLY the provided context.

STRICT RULES:
1. Answer ONLY based on the information in the provided context below. Do not use outside knowledge.
2. If the context does not contain sufficient information to answer the question, respond with: "I don't know based on the provided research papers."
3. Do NOT make up or hallucinate any information, paper titles, citations, or page numbers not present in the context.
4. Always cite your sources by referencing the paper title and page number.
5. Be detailed and technical in your responses — this is for a research audience.
6. Use the chat history to understand the context of follow-up questions, but always ground your answers in the retrieved context.

CONTEXT (Retrieved from research papers):
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Provide a comprehensive answer with citations. After your answer, list the top sources used in this exact format:

**Sources Used:**
1. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
2. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
3. **[Paper Title]** — Page [X]: "[Brief relevant excerpt from the passage]"
"""


# =============================================================================
# 4. RAG Chain Construction (LCEL)
# =============================================================================

def build_rag_chain(retriever, llm=None):
    """
    Build the core RAG chain using LCEL (LangChain Expression Language).

    Pipeline flow:
        question → retriever → format_docs → prompt → Gemini LLM → parse

    The chain returns the LLM's response as a string. Source documents
    are tracked separately via the retriever for citation display.

    Args:
        retriever: LangChain retriever (any strategy from retrieval.py).
        llm: ChatGoogleGenerativeAI instance. Defaults to config settings.

    Returns:
        LCEL chain that accepts a question string and returns an answer.
    """
    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # LCEL chain composition
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs_with_citations),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[+] RAG chain built (LCEL pipeline with Gemini)")
    return rag_chain


# =============================================================================
# 5. Query Interface (with source tracking)
# =============================================================================

def query_rag(
    question: str,
    retriever,
    llm=None,
    top_k_sources: int = 3,
    answer_mode: str = "standard",
) -> dict:
    """
    Query the RAG pipeline and return answer with source citations.
    """
    import time
    from src.advanced_features import calculate_grounding_confidence

    t_start = time.time()
    if llm is None:
        llm = get_llm()

    # Retrieve documents
    from src.retrieval import RerankedRetriever
    t_ret_start = time.time()
    retrieved_docs = retriever.invoke(question)
    t_ret_end = time.time()

    # Format context for LLM
    context = format_docs_with_citations(retrieved_docs)
    sources = extract_source_info(retrieved_docs, top_k=top_k_sources)
    grounding = calculate_grounding_confidence(retrieved_docs)

    # Check for empty retrieval / low confidence protection
    if not retrieved_docs:
        return {
            "question": question,
            "answer": "I don't know based on the provided research papers.",
            "sources": [],
            "retrieved_docs": [],
            "grounding": grounding,
            "latency": {
                "retrieval_time": round(t_ret_end - t_ret_start, 3),
                "generation_time": 0.0,
                "total_time": round(time.time() - t_start, 3),
            },
        }

    t_gen_start = time.time()
    try:
        # Check answer mode prompt customizer
        mode_instruction = ANSWER_MODE_INSTRUCTIONS.get(answer_mode.lower(), ANSWER_MODE_INSTRUCTIONS["standard"])
        mode_prompt_template = RAG_PROMPT_TEMPLATE + f"\n\nFORMAT INSTRUCTION: {mode_instruction}"

        prompt = ChatPromptTemplate.from_template(mode_prompt_template)
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": question,
        })
    except Exception as e:
        answer = (
            "⚠️ **Gemini API Request Failed**\n\n"
            "Please check your `GEMINI_API_KEY`, API quota, model availability, and internet connection.\n\n"
            "*(Displaying exact relevant passages retrieved directly from the indexed vector store below)*\n\n"
        )
        for i, src in enumerate(sources, 1):
            answer += f"**Excerpt {i} [{src['paper_title']} — Page {src['page']}]:**\n\n> \"{src['passage']}\"\n\n"

    t_gen_end = time.time()

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
        "grounding": grounding,
        "latency": {
            "retrieval_time": round(t_ret_end - t_ret_start, 3),
            "generation_time": round(t_gen_end - t_gen_start, 3),
            "total_time": round(t_gen_end - t_start, 3),
        },
    }


def display_rag_response(result: dict) -> None:
    """
    Pretty-print a RAG response with sources.

    Args:
        result: Dictionary returned by query_rag().
    """
    print("=" * 80)
    print(f"❓ QUESTION: {result['question']}")
    print("=" * 80)
    print(f"\n💬 ANSWER:\n{result['answer']}")
    print("\n" + "-" * 60)
    print("📚 TOP-3 SUPPORTING SOURCES:")
    print("-" * 60)
    for i, source in enumerate(result["sources"], 1):
        print(f"\n  [{i}] {source['paper_title']} — Page {source['page']}")
        print(f"      File: {source['source_file']}")
        print(f"      Passage: \"{source['passage'][:200]}...\"")
    print("\n" + "=" * 80)
