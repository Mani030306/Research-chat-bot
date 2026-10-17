"""
Memory Module — Stretch Goal 1: Multi-User Conversational RAG.

Adds chat history / memory for multi-turn Q&A. Uses
ConversationBufferWindowMemory to maintain a sliding window of
recent exchanges, allowing follow-up questions that reference
prior context.

Example conversation:
  User: "What is self-attention?"
  Bot:  "Self-attention is... [from Attention Is All You Need]"
  User: "How does BERT use it differently?"  ← follow-up
  Bot:  "BERT uses self-attention bidirectionally... [from BERT paper]"

Design Decision: We use a sliding window (k=5) rather than full
history to prevent context overflow. For longer conversations,
ConversationSummaryBufferMemory could be used instead to
summarize older turns.
"""

from typing import Optional

try:
    from langchain_classic.memory import ConversationBufferWindowMemory
except ImportError:
    try:
        from langchain.memory import ConversationBufferWindowMemory
    except ImportError:
        from langchain_community.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from src.config import CONVERSATION_WINDOW_SIZE
from src.rag_chain import (
    get_llm,
    format_docs_with_citations,
    extract_source_info,
    CONVERSATIONAL_RAG_PROMPT_TEMPLATE,
)


# =============================================================================
# 1. Conversation Memory
# =============================================================================

def create_memory(window_size: int = CONVERSATION_WINDOW_SIZE) -> ConversationBufferWindowMemory:
    """
    Create a conversation memory with sliding window.

    Keeps the last `window_size` exchanges (user + AI pairs).
    Older exchanges are dropped to prevent context overflow.

    Args:
        window_size: Number of past exchanges to retain (default: 5).

    Returns:
        ConversationBufferWindowMemory instance.
    """
    memory = ConversationBufferWindowMemory(
        k=window_size,
        memory_key="chat_history",
        return_messages=False,  # Return as string for prompt injection
        input_key="question",
        output_key="answer",
    )
    print(f"[+] Conversation memory created (window_size={window_size})")
    return memory


# =============================================================================
# 2. Conversational RAG Chain
# =============================================================================

class ConversationalRAG:
    """
    RAG pipeline with multi-turn conversational memory.

    Wraps the base RAG chain with ConversationBufferWindowMemory,
    allowing follow-up questions that reference prior conversation
    context. Each exchange is saved to memory after generation.
    """

    def __init__(
        self,
        retriever,
        llm=None,
        memory: Optional[ConversationBufferWindowMemory] = None,
    ):
        """
        Initialize the conversational RAG pipeline.

        Args:
            retriever: LangChain retriever instance.
            llm: ChatOpenAI instance. Defaults to config settings.
            memory: Conversation memory. Creates new if None.
        """
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.memory = memory or create_memory()
        self.prompt = ChatPromptTemplate.from_template(
            CONVERSATIONAL_RAG_PROMPT_TEMPLATE
        )

        print("[+] Conversational RAG pipeline initialized")

    def query(
        self,
        question: str,
        top_k_sources: int = 3,
        answer_mode: str = "standard",
    ) -> dict:
        """
        Ask a question with conversation context.
        """
        import time
        from src.advanced_features import calculate_grounding_confidence
        from src.rag_chain import ANSWER_MODE_INSTRUCTIONS

        t_start = time.time()
        from src.retrieval import RerankedRetriever

        # Retrieve relevant documents
        t_ret_start = time.time()
        retrieved_docs = self.retriever.invoke(question)
        t_ret_end = time.time()

        # Format context
        context = format_docs_with_citations(retrieved_docs)
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        sources = extract_source_info(retrieved_docs, top_k=top_k_sources)
        grounding = calculate_grounding_confidence(retrieved_docs)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I don't know based on the provided research papers.",
                "sources": [],
                "retrieved_docs": [],
                "chat_history": chat_history,
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
            if answer_mode.lower() in ANSWER_MODE_INSTRUCTIONS and answer_mode.lower() != "standard":
                mode_inst = ANSWER_MODE_INSTRUCTIONS[answer_mode.lower()]
                custom_prompt_text = CONVERSATIONAL_RAG_PROMPT_TEMPLATE + f"\n\nFORMAT INSTRUCTION: {mode_inst}"
                custom_prompt = ChatPromptTemplate.from_template(custom_prompt_text)
                chain = custom_prompt | self.llm | StrOutputParser()
            else:
                chain = self.prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "chat_history": chat_history if chat_history else "No previous conversation.",
                "question": question,
            })
            # Save to memory
            self.memory.save_context(
                {"question": question},
                {"answer": answer},
            )
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
            "chat_history": chat_history,
            "grounding": grounding,
            "latency": {
                "retrieval_time": round(t_ret_end - t_ret_start, 3),
                "generation_time": round(t_gen_end - t_gen_start, 3),
                "total_time": round(t_gen_end - t_start, 3),
            },
        }

    def clear_history(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        print("[+] Conversation history cleared")

    def get_history(self) -> str:
        """Get the current conversation history as a string."""
        return self.memory.load_memory_variables({}).get("chat_history", "")


# =============================================================================
# 3. Demo Function
# =============================================================================

def demo_conversational_rag(retriever, llm=None) -> None:
    """
    Demonstrate multi-turn conversational RAG with sample exchanges.

    Shows how follow-up questions reference prior context.

    Args:
        retriever: LangChain retriever instance.
        llm: ChatOpenAI instance.
    """
    conv_rag = ConversationalRAG(retriever=retriever, llm=llm)

    demo_questions = [
        "What is the self-attention mechanism in transformers?",
        "How does BERT use self-attention differently from the original transformer?",
        "Can you compare this with how GPT-3 uses attention?",
    ]

    print("\n" + "=" * 80)
    print("CONVERSATIONAL RAG DEMO")
    print("Demonstrating multi-turn Q&A with memory")
    print("=" * 80)

    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i}")
        print(f"{'─' * 60}")

        result = conv_rag.query(question)

        print(f"👤 User: {result['question']}")
        print(f"\n🤖 Assistant: {result['answer']}")
        print(f"\n📚 Sources:")
        for j, source in enumerate(result["sources"], 1):
            print(f"  [{j}] {source['paper_title']} — Page {source['page']}")

        if result["chat_history"]:
            print(f"\n💭 Chat History Length: {len(result['chat_history'])} chars")

    print("\n[+] Conversational RAG demo complete")
