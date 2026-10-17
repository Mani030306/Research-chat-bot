"""
Streamlit Application — Research Paper Answer Bot (Enhanced).

A production-quality chat interface for the Research Paper Answer Bot
with the following features:

- Chat interface with streaming responses
- Expandable source viewer showing top-3 citations
- Conversational memory persisted across the session
- Retrieval strategy selector
- Answer mode selector (Standard, Concise, Detailed, Viva)
- Recent questions history with click-to-reuse
- Advanced source explorer with full passages & scores
- Performance metrics & latency transparency
- Grounding confidence badge
- Smart follow-up suggestions
- Response feedback (thumbs up/down)
- Conversation export (Markdown & TXT)
- Research paper explorer with chunk stats
- System info sidebar (papers indexed, chunks, model info)
- "Clear Chat" button to reset conversation

Run with: streamlit run streamlit_app.py
"""

import sys
import os
import time

sys.modules['tensorflow'] = None
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.advanced_features import (
    generate_smart_followup_questions,
    export_chat_history,
    get_paper_library_stats,
)


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS for polished appearance
# =============================================================================
st.markdown("""
<style>
    /* Main container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Chat message styling */
    .source-card {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }

    .dark .source-card {
        background-color: #262730;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }

    /* Sidebar stats */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: center;
    }

    /* Grounding badges */
    .grounding-strong {
        background-color: #d4edda; color: #155724;
        padding: 6px 14px; border-radius: 20px;
        font-weight: 600; display: inline-block; margin: 4px 0;
    }
    .grounding-moderate {
        background-color: #fff3cd; color: #856404;
        padding: 6px 14px; border-radius: 20px;
        font-weight: 600; display: inline-block; margin: 4px 0;
    }
    .grounding-weak {
        background-color: #f8d7da; color: #721c24;
        padding: 6px 14px; border-radius: 20px;
        font-weight: 600; display: inline-block; margin: 4px 0;
    }

    /* Follow-up suggestion buttons */
    .followup-btn {
        text-align: left;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "pipeline_initialized": False,
        "conv_rag": None,
        "vectorstore": None,
        "chunks": None,
        "retrieval_strategy": "hybrid",
        "num_papers": 0,
        "num_chunks": 0,
        # Advanced feature state
        "answer_mode": "standard",
        "recent_questions": [],
        "feedback": {},          # {msg_index: "up"/"down"}
        "pending_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# Pipeline Initialization
# =============================================================================

@st.cache_resource
def load_pipeline():
    """
    Load and cache the RAG pipeline components.

    This runs once and caches the vector store, chunks, and
    conversational RAG instance across Streamlit reruns.
    """
    try:
        from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_HF, CHROMA_COLLECTION_OPENAI
        from src.embeddings import get_hf_embeddings, get_openai_embeddings
        from src.vectorstore import load_vectorstore, setup_hf_vectorstore
        from src.chunking import chunk_recursive
        from src.data_loader import load_papers_pypdf, download_papers

        # Load HuggingFace embeddings (local & free)
        embedding_model = get_hf_embeddings()

        # Load papers & chunks
        download_papers()
        documents = load_papers_pypdf()
        chunks = chunk_recursive(documents)

        # Load or setup HF vector store
        vectorstore = setup_hf_vectorstore(chunks)

        return vectorstore, chunks, embedding_model
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        raise


def get_retriever(vectorstore, chunks, strategy: str):
    """Get a retriever based on the selected strategy."""
    from src.retrieval import (
        get_dense_retriever,
        get_mmr_retriever,
        get_hybrid_retriever,
        get_reranked_retriever,
        get_multiquery_retriever,
    )

    strategy_map = {
        "dense": lambda: get_dense_retriever(vectorstore),
        "mmr": lambda: get_mmr_retriever(vectorstore),
        "hybrid": lambda: get_hybrid_retriever(vectorstore, chunks),
        "reranker": lambda: get_reranked_retriever(vectorstore),
        "multi_query": lambda: get_multiquery_retriever(vectorstore),
    }

    return strategy_map.get(strategy, strategy_map["hybrid"])()


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar with controls and system info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # ---- Retrieval strategy selector ----
        strategy = st.selectbox(
            "Retrieval Strategy",
            options=["hybrid", "dense", "mmr", "reranker", "multi_query"],
            index=0,
            format_func=lambda x: {
                "dense": "🔵 Dense (Cosine Similarity)",
                "mmr": "🟣 MMR (Max Marginal Relevance)",
                "hybrid": "🟢 Hybrid (BM25 + Dense)",
                "reranker": "🟠 Cross-Encoder Reranker",
                "multi_query": "🔴 Multi-Query (LLM-generated)",
            }.get(x, x),
            help="Select the retrieval strategy for finding relevant passages.",
        )
        st.session_state.retrieval_strategy = strategy

        # ---- Answer mode selector ----
        answer_mode = st.selectbox(
            "Answer Mode",
            options=["standard", "concise", "detailed", "viva"],
            index=0,
            format_func=lambda x: {
                "standard": "📖 Standard",
                "concise": "✂️ Concise",
                "detailed": "🔎 Detailed",
                "viva": "🎓 Technical / Viva",
            }.get(x, x),
            help="Choose how Gemini should format the answer.",
        )
        st.session_state.answer_mode = answer_mode

        st.markdown("---")

        # ---- Recent Questions History ----
        st.markdown("## 📜 Recent Questions")
        if st.session_state.recent_questions:
            # Show last 10, most recent first
            for idx, q in enumerate(reversed(st.session_state.recent_questions[-10:]), 1):
                truncated = q[:60] + "..." if len(q) > 60 else q
                if st.button(f"🔁 {truncated}", key=f"recent_q_{idx}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
            if st.button("🗑️ Clear Query History", key="clear_q_hist", use_container_width=True):
                st.session_state.recent_questions = []
                st.rerun()
        else:
            st.caption("No questions asked yet this session.")

        st.markdown("---")

        # ---- Clear chat button ----
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.recent_questions = []
            st.session_state.feedback = {}
            if st.session_state.conv_rag:
                st.session_state.conv_rag.clear_history()
            st.rerun()

        st.markdown("---")

        # ---- Research Paper Explorer ----
        st.markdown("## 📚 Research Paper Explorer")
        if st.session_state.pipeline_initialized and st.session_state.chunks:
            paper_stats = get_paper_library_stats(st.session_state.chunks)
            for paper in paper_stats:
                with st.expander(f"📄 {paper['paper_title']}", expanded=False):
                    st.markdown(
                        f"- **Chunks:** {paper['chunk_count']}\n"
                        f"- **Pages:** {paper['page_count']} (pp. {paper['page_range']})\n"
                    )
        else:
            st.caption("Load the pipeline to explore papers.")

        st.markdown("---")

        # ---- System info ----
        st.markdown("## 📊 System Info")

        if st.session_state.pipeline_initialized:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", st.session_state.num_papers)
            with col2:
                st.metric("Chunks", st.session_state.num_chunks)

            st.markdown("""
            **LLM:** Google Gemini (gemini-2.5-flash)
            **Embedding Model:** HuggingFace BGE-small (BAAI/bge-small-en-v1.5)
            **Vector DB:** ChromaDB (local)
            **Memory:** Sliding window (k=5)
            """)
        else:
            st.info("Pipeline not yet initialized.")

        st.markdown("---")

        # ---- Export Conversation ----
        st.markdown("## 📥 Export Conversation")
        if st.session_state.messages:
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                md_export = export_chat_history(st.session_state.messages, "markdown")
                st.download_button(
                    "⬇️ .md",
                    data=md_export,
                    file_name=f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with exp_col2:
                txt_export = export_chat_history(st.session_state.messages, "txt")
                st.download_button(
                    "⬇️ .txt",
                    data=txt_export,
                    file_name=f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.caption("Start a conversation to enable export.")

        st.markdown("---")
        st.markdown(
            "### 📝 About\n"
            "**Research Paper Answer Bot**\n\n"
            "A RAG chatbot built over seminal GenAI/LLM research papers. "
            "Ask questions about transformers, BERT, GPT-3, LoRA, RAG, "
            "InstructGPT, T5, and LLaMA.\n\n"
            "*GenAI Pinnacle Plus Program — Capstone Project*"
        )


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Research Paper Answer Bot</h1>
        <p style="color: #666; font-size: 1.1em;">
            Ask questions about seminal GenAI &amp; LLM research papers
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(
                            f"**{i}. {source['paper_title']}** — Page {source['page']}\n\n"
                            f"> {source['passage'][:300]}..."
                        )

                # Show grounding badge for historical messages
                if "grounding" in message:
                    _render_grounding_badge(message["grounding"])

                # Show latency for historical messages
                if "latency" in message:
                    _render_latency_table(message["latency"], message.get("retrieved_count", 0))

    # ---- Determine prompt (from chat_input or pending_question) ----
    prompt = st.chat_input("Ask a question about the research papers...")

    # Check for pending question from sidebar click
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if prompt:
        # Track recent questions
        if prompt not in st.session_state.recent_questions:
            st.session_state.recent_questions.append(prompt)

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.conv_rag is None:
                        st.error("Pipeline not initialized. Please wait for setup to complete.")
                        return

                    # Pass answer_mode to query
                    result = st.session_state.conv_rag.query(
                        prompt,
                        answer_mode=st.session_state.answer_mode,
                    )

                    # ---- Display answer ----
                    st.markdown(result["answer"])

                    # ---- Top-3 Sources (original behavior preserved) ----
                    with st.expander("📚 View Sources", expanded=True):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(
                                f"**{i}. {source['paper_title']}** — Page {source['page']}\n\n"
                                f"> {source['passage'][:300]}..."
                            )

                    # ---- Grounding Confidence Badge ----
                    grounding = result.get("grounding", {})
                    _render_grounding_badge(grounding)

                    # ---- Performance Metrics ----
                    latency = result.get("latency", {})
                    retrieved_count = len(result.get("retrieved_docs", []))
                    _render_latency_table(latency, retrieved_count)

                    # ---- Advanced Source Explorer ----
                    retrieved_docs = result.get("retrieved_docs", [])
                    if retrieved_docs:
                        with st.expander("🔍 Advanced Source Explorer", expanded=False):
                            for idx, doc in enumerate(retrieved_docs, 1):
                                meta = doc.metadata
                                title = meta.get("paper_title", "Unknown")
                                page = meta.get("page", "?")
                                score = meta.get("relevance_score", meta.get("score", "N/A"))
                                if isinstance(score, float):
                                    score = f"{score:.4f}"
                                st.markdown(f"---")
                                st.markdown(
                                    f"**Chunk {idx}** | 📄 **{title}** — Page {page} | "
                                    f"Score: `{score}`"
                                )
                                st.markdown(
                                    f"<details><summary>View full passage</summary>"
                                    f"<p style='font-size:0.85em; color:#555;'>{doc.page_content}</p>"
                                    f"</details>",
                                    unsafe_allow_html=True,
                                )

                    # ---- Smart Follow-Up Suggestions ----
                    suggestions = generate_smart_followup_questions(
                        prompt, result["answer"], result.get("retrieved_docs", [])
                    )
                    if suggestions:
                        st.markdown("**💡 Follow-up suggestions:**")
                        suggestion_cols = st.columns(len(suggestions))
                        for s_idx, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
                            with col:
                                if st.button(
                                    suggestion[:80],
                                    key=f"followup_{len(st.session_state.messages)}_{s_idx}",
                                    use_container_width=True,
                                ):
                                    st.session_state.pending_question = suggestion
                                    st.rerun()

                    # ---- Response Feedback ----
                    msg_index = len(st.session_state.messages)
                    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
                    with fb_col1:
                        if st.button("👍", key=f"fb_up_{msg_index}"):
                            st.session_state.feedback[msg_index] = "up"
                            st.toast("Thanks for your feedback! 👍")
                    with fb_col2:
                        if st.button("👎", key=f"fb_down_{msg_index}"):
                            st.session_state.feedback[msg_index] = "down"
                            st.toast("Thanks for your feedback! We'll improve. 👎")

                    # ---- Save to session state ----
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "grounding": grounding,
                        "latency": latency,
                        "retrieved_count": retrieved_count,
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# =============================================================================
# UI Helper Functions
# =============================================================================

def _render_grounding_badge(grounding: dict):
    """Render a colored grounding confidence badge."""
    if not grounding:
        return
    level = grounding.get("level", "weak")
    status = grounding.get("status", "")
    explanation = grounding.get("explanation", "")
    css_class = f"grounding-{level}"
    st.markdown(
        f'<span class="{css_class}">{status}</span> '
        f'<span style="font-size:0.82em; color:#888;">{explanation}</span>',
        unsafe_allow_html=True,
    )


def _render_latency_table(latency: dict, retrieved_count: int = 0):
    """Render a compact performance metrics expander."""
    if not latency:
        return
    with st.expander("⏱️ Performance Metrics", expanded=False):
        perf_cols = st.columns(4)
        perf_cols[0].metric("Retrieval", f"{latency.get('retrieval_time', 0):.2f}s")
        perf_cols[1].metric("Generation", f"{latency.get('generation_time', 0):.2f}s")
        perf_cols[2].metric("Total", f"{latency.get('total_time', 0):.2f}s")
        perf_cols[3].metric("Chunks Retrieved", retrieved_count)


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    init_session_state()
    render_sidebar()

    # Initialize pipeline
    if not st.session_state.pipeline_initialized:
        with st.spinner("🚀 Initializing RAG pipeline... This may take a minute on first run."):
            try:
                vectorstore, chunks, embedding_model = load_pipeline()
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks
                st.session_state.num_papers = len(set(
                    c.metadata.get("paper_title", "") for c in chunks
                ))
                st.session_state.num_chunks = len(chunks)

                # Create retriever and conversational RAG
                retriever = get_retriever(
                    vectorstore, chunks, st.session_state.retrieval_strategy
                )

                from src.memory import ConversationalRAG
                st.session_state.conv_rag = ConversationalRAG(
                    retriever=retriever,
                )

                st.session_state.pipeline_initialized = True
                st.rerun()

            except Exception as e:
                st.error(
                    f"Failed to initialize pipeline: {e}\n\n"
                    "Make sure you have:\n"
                    "1. Created a `.env` file with your `GEMINI_API_KEY`\n"
                    "2. Run the notebook first to download papers and build the vector store\n"
                    "3. Installed all dependencies: `pip install -r requirements.txt`"
                )
                return
    else:
        # Update retriever if strategy changed
        if st.session_state.vectorstore and st.session_state.chunks:
            current_strategy = st.session_state.retrieval_strategy
            retriever = get_retriever(
                st.session_state.vectorstore,
                st.session_state.chunks,
                current_strategy,
            )
            if st.session_state.conv_rag:
                st.session_state.conv_rag.retriever = retriever

    render_chat()


if __name__ == "__main__":
    main()
