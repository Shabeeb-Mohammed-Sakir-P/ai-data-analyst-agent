from llm_client import call_llm


def build_context_summary(
    profiling_findings: dict,
    hypotheses: list,
    test_results: list,
    report: str,
) -> str:
    """
    Packages everything the pipeline has learned about the dataset into
    one context block the QA agent can reference when answering questions.
    """
    return f"""=== Full Analysis Report ===
{report}

=== Raw Profiling Findings ===
{profiling_findings}

=== Hypotheses Tested ===
{hypotheses}

=== Statistical Test Results ===
{test_results}
"""


def answer_question(question: str, context: str, chat_history: list = None) -> str:
    """
    Answers a user's question about the analysis, grounded strictly in the
    provided context. If the answer isn't in the context, says so honestly
    instead of guessing.
    """
    if chat_history is None:
        chat_history = []

    history_text = ""
    if chat_history:
        history_text = "\n\n=== Previous conversation ===\n"
        for turn in chat_history:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""You are a helpful data analyst assistant. Answer the user's
question using ONLY the analysis context below. If the answer cannot be
determined from this context, say so clearly rather than guessing or making
up information. Be concise and direct.

{context}
{history_text}

User's question: {question}

Answer:"""

    return call_llm(prompt)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    import pandas as pd
    from profiling_agent import analyze_dataset
    from cleaning_agent import apply_cleaning_action, propose_cleaning_actions
    from hypothesis_agent import generate_hypotheses
    from statistical_testing_agent import run_statistical_test
    from report_agent import generate_report
    from visualization_agent import decide_chart_types
    from feature_engineering_agent import analyze_for_feature_engineering, propose_feature_engineering_actions

    print("Running full pipeline to build context (this takes a bit)...\n")

    profiling_findings = analyze_dataset("data/sample_messy_customers.csv")
    cleaning_actions = propose_cleaning_actions(profiling_findings)

    df = pd.read_csv("data/sample_messy_customers.csv")
    df = apply_cleaning_action(df, {"action": "remove_duplicates", "column": "all"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "region"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "is_active"})
    df = apply_cleaning_action(df, {"action": "fix_dtype", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "monthly_spend"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "region"})
    df = df[df["monthly_spend"] < 5000]

    hypotheses = generate_hypotheses(df)
    test_results = [run_statistical_test(df, h) for h in hypotheses]
    chart_specs = decide_chart_types(test_results, list(df.columns))
    fe_analysis = analyze_for_feature_engineering(df)
    fe_actions = propose_feature_engineering_actions(fe_analysis)

    report = generate_report(
        profiling_findings, cleaning_actions, hypotheses,
        test_results, chart_specs, fe_actions
    )

    context = build_context_summary(profiling_findings, hypotheses, test_results, report)

    print("Pipeline complete. You can now ask questions.")
    print("Type 'quit' to exit.\n")

    chat_history = []
    while True:
        question = input("Your question: ")
        if question.lower() in ("quit", "exit"):
            break

        answer = answer_question(question, context, chat_history)
        print(f"\nAssistant: {answer}\n")

        chat_history.append({"question": question, "answer": answer})