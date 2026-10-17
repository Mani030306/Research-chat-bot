"""
Evaluation Module — Step 7: Testing & Evaluation.

Evaluates the RAG pipeline on 10 test questions spanning different
research papers. Uses both manual inspection and automated evaluation
via RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision,
Context Recall).

The evaluation serves three purposes:
1. Validate that the pipeline produces accurate, grounded answers
2. Identify failure cases and retrieval gaps
3. Provide quantitative evidence for the final rubric evaluation
"""

from typing import Optional
import pandas as pd

from langchain_core.documents import Document

from src.config import OPENAI_API_KEY


# =============================================================================
# 1. Test Questions
# =============================================================================

TEST_QUESTIONS = [
    {
        "question": "What is the self-attention mechanism in transformers and how does it compute attention weights?",
        "expected_papers": ["Attention Is All You Need"],
        "difficulty": "medium",
    },
    {
        "question": "How does BERT's masked language model (MLM) pre-training objective work?",
        "expected_papers": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
        "difficulty": "medium",
    },
    {
        "question": "What is few-shot learning as described in the GPT-3 paper, and how does it differ from fine-tuning?",
        "expected_papers": ["Language Models are Few-Shot Learners (GPT-3)"],
        "difficulty": "medium",
    },
    {
        "question": "How does T5 frame all NLP tasks as text-to-text problems?",
        "expected_papers": ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)"],
        "difficulty": "medium",
    },
    {
        "question": "What problem does LoRA solve, and how does it reduce the number of trainable parameters during fine-tuning?",
        "expected_papers": ["LoRA: Low-Rank Adaptation of Large Language Models"],
        "difficulty": "medium",
    },
    {
        "question": "How does Retrieval-Augmented Generation (RAG) combine a retriever with a generator?",
        "expected_papers": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
        "difficulty": "medium",
    },
    {
        "question": "What is RLHF (Reinforcement Learning from Human Feedback) and how was it used in InstructGPT?",
        "expected_papers": ["Training Language Models to Follow Instructions with Human Feedback (InstructGPT)"],
        "difficulty": "hard",
    },
    {
        "question": "What architectural choices make LLaMA more efficient than previous large language models?",
        "expected_papers": ["LLaMA: Open and Efficient Foundation Language Models"],
        "difficulty": "hard",
    },
    {
        "question": "Compare the pre-training objectives of BERT and GPT — how do they differ and what are the implications?",
        "expected_papers": [
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "Language Models are Few-Shot Learners (GPT-3)",
        ],
        "difficulty": "hard",
    },
    {
        "question": "What are the key differences between encoder-only, decoder-only, and encoder-decoder transformer architectures?",
        "expected_papers": [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
        ],
        "difficulty": "hard",
    },
]


# =============================================================================
# 2. Run Evaluation
# =============================================================================

def run_evaluation(
    retriever,
    llm=None,
    questions: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """
    Run the RAG pipeline on all test questions and collect results.

    For each question:
    - Retrieves relevant chunks
    - Generates an answer with citations
    - Checks if expected papers appear in retrieved sources
    - Records results for analysis

    Args:
        retriever: LangChain retriever instance.
        llm: ChatOpenAI instance.
        questions: List of test question dicts. Defaults to TEST_QUESTIONS.

    Returns:
        DataFrame with evaluation results.
    """
    from src.rag_chain import query_rag, display_rag_response

    questions = questions or TEST_QUESTIONS
    results = []

    print("\n" + "=" * 80)
    print("RAG PIPELINE EVALUATION")
    print(f"Running {len(questions)} test questions...")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'─' * 60}")

        try:
            result = query_rag(
                question=q["question"],
                retriever=retriever,
                llm=llm,
            )

            display_rag_response(result)

            # Check if expected papers were retrieved
            retrieved_papers = [s["paper_title"] for s in result["sources"]]
            expected_papers = q.get("expected_papers", [])
            papers_found = sum(
                1 for ep in expected_papers
                if any(ep.lower() in rp.lower() for rp in retrieved_papers)
            )
            retrieval_hit = papers_found > 0

            results.append({
                "question_id": i,
                "question": q["question"],
                "difficulty": q.get("difficulty", "medium"),
                "answer_length": len(result["answer"]),
                "num_sources": len(result["sources"]),
                "expected_papers": ", ".join(expected_papers),
                "retrieved_papers": ", ".join(retrieved_papers),
                "retrieval_hit": retrieval_hit,
                "answer_preview": result["answer"][:200] + "...",
            })

        except Exception as e:
            print(f"  ⚠ Error: {e}")
            results.append({
                "question_id": i,
                "question": q["question"],
                "difficulty": q.get("difficulty", "medium"),
                "answer_length": 0,
                "num_sources": 0,
                "expected_papers": ", ".join(q.get("expected_papers", [])),
                "retrieved_papers": "",
                "retrieval_hit": False,
                "answer_preview": f"ERROR: {str(e)}",
            })

    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions: {len(df)}")
    print(f"  Retrieval hits: {df['retrieval_hit'].sum()}/{len(df)} "
          f"({df['retrieval_hit'].mean()*100:.0f}%)")
    print(f"  Avg answer length: {df['answer_length'].mean():.0f} chars")
    print(f"  Questions with sources: {(df['num_sources'] > 0).sum()}/{len(df)}")

    return df


# =============================================================================
# 3. RAGAS Automated Evaluation
# =============================================================================

def run_ragas_evaluation(
    retriever,
    llm=None,
    questions: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """
    Run automated evaluation using RAGAS metrics.

    RAGAS Metrics:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are relevant chunks ranked higher?
    - Context Recall: Did we retrieve all necessary information?

    Note: RAGAS uses an LLM (GPT-4) as a judge. This incurs API costs.

    Args:
        retriever: LangChain retriever instance.
        llm: ChatOpenAI instance.
        questions: List of test question dicts.

    Returns:
        DataFrame with RAGAS scores per question.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("⚠ RAGAS or datasets not installed. Skipping automated evaluation.")
        print("  Install with: pip install ragas datasets")
        return pd.DataFrame()

    from src.rag_chain import query_rag, format_docs_with_citations

    questions = questions or TEST_QUESTIONS

    # Collect data for RAGAS
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print("\nCollecting data for RAGAS evaluation...")
    for q in questions:
        try:
            result = query_rag(
                question=q["question"],
                retriever=retriever,
                llm=llm,
            )
            ragas_data["question"].append(q["question"])
            ragas_data["answer"].append(result["answer"])
            ragas_data["contexts"].append(
                [doc.page_content for doc in result["retrieved_docs"]]
            )
            # Use expected papers as a rough ground truth reference
            ragas_data["ground_truth"].append(
                f"The answer should reference: {', '.join(q.get('expected_papers', ['relevant papers']))}"
            )
        except Exception as e:
            print(f"  ⚠ Skipping question due to error: {e}")

    if not ragas_data["question"]:
        print("⚠ No data collected for RAGAS. Skipping.")
        return pd.DataFrame()

    # Create RAGAS dataset
    dataset = Dataset.from_dict(ragas_data)

    # Run evaluation
    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        print("\n" + "=" * 60)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 60)

        # Convert to DataFrame
        ragas_df = ragas_result.to_pandas()
        print(ragas_df.to_string())

        # Print averages
        print("\nAverage Scores:")
        for col in ragas_df.columns:
            if col not in ["question", "answer", "contexts", "ground_truth"]:
                print(f"  {col}: {ragas_df[col].mean():.4f}")

        return ragas_df

    except Exception as e:
        print(f"⚠ RAGAS evaluation failed: {e}")
        print("  This is common with newer RAGAS versions. Results from manual evaluation above are sufficient.")
        return pd.DataFrame()


# =============================================================================
# 4. Failure Analysis
# =============================================================================

def analyze_failures(eval_df: pd.DataFrame) -> None:
    """
    Analyze and document failure cases from the evaluation.

    Identifies questions where:
    - Expected papers were not retrieved
    - Answers were too short (potential "I don't know" responses)
    - No sources were found

    Args:
        eval_df: DataFrame from run_evaluation().
    """
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    # Retrieval failures
    retrieval_failures = eval_df[~eval_df["retrieval_hit"]]
    if len(retrieval_failures) > 0:
        print(f"\n❌ Retrieval Failures ({len(retrieval_failures)}):")
        for _, row in retrieval_failures.iterrows():
            print(f"  Q{row['question_id']}: {row['question'][:80]}...")
            print(f"    Expected: {row['expected_papers']}")
            print(f"    Got: {row['retrieved_papers']}")
    else:
        print("\n✓ No retrieval failures — all expected papers were found.")

    # Short answers (potential "I don't know")
    short_answers = eval_df[eval_df["answer_length"] < 100]
    if len(short_answers) > 0:
        print(f"\n⚠ Short/Uncertain Answers ({len(short_answers)}):")
        for _, row in short_answers.iterrows():
            print(f"  Q{row['question_id']}: {row['answer_preview'][:120]}")

    # Overall assessment
    hit_rate = eval_df["retrieval_hit"].mean() * 100
    print(f"\n📊 Overall Retrieval Hit Rate: {hit_rate:.0f}%")
    if hit_rate >= 80:
        print("  ✓ Good retrieval performance.")
    elif hit_rate >= 60:
        print("  ⚠ Moderate retrieval performance. Consider tuning retrieval strategy.")
    else:
        print("  ❌ Poor retrieval performance. Review chunking and embedding choices.")
