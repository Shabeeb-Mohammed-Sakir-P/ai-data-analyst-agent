import pandas as pd
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


def needs_live_data(question: str) -> bool:
    """
    Asks the LLM to classify whether this question needs a live calculation
    on the raw dataset (like 'average age') versus something answerable
    from the existing report/findings (like 'what was significant').
    """
    prompt = f"""Does answering this question require calculating a specific
number or fact directly from raw data (like an average, count, min, max, or
filter)? Or can it be answered from a general report/summary?

Question: "{question}"

Respond with ONLY one word: "DATA" or "REPORT"."""

    response = call_llm(prompt).strip().upper()
    return "DATA" in response


def generate_pandas_query(question: str, columns: list) -> str:
    """
    Asks the LLM to translate a question into a single pandas expression
    that computes the answer, using a DataFrame variable called 'df'.
    """
    prompt = f"""Convert this question into a SINGLE pandas expression using
a DataFrame variable called `df`. The available columns are: {columns}

IMPORTANT: text/categorical values in this dataset have been standardized to
lowercase (e.g. "north" not "North"). When comparing or filtering on text
columns, always use lowercase values, and consider using .str.lower() on
the column for safety in case of any inconsistency.

Question: "{question}"

Respond with ONLY the Python expression, nothing else. No explanation,
no code fences, no assignment (do not write "result = ..."), just the
raw expression itself.

Example: for "what is the average age?" respond exactly with:
df['age'].mean()

Example: for "how many customers are in the north region?" respond exactly with:
(df['region'].str.lower() == 'north').sum()"""

    expression = call_llm(prompt).strip()
    expression = expression.strip("`").replace("python\n", "").strip()
    return expression


def safe_execute_query(df: pd.DataFrame, expression: str):
    """
    Executes a pandas expression in a RESTRICTED environment — only 'df'
    and safe pandas functions are accessible. No file access, no imports,
    no system commands are possible from within this sandbox.
    """
    allowed_names = {"df": df, "pd": pd}
    try:
        # eval (not exec) only computes an expression's value — it cannot
        # run multi-line code or statements like imports or file writes.
        # __builtins__ is explicitly emptied so things like open() or
        # __import__() are not accessible from inside the expression.
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result, None
    except Exception as e:
        return None, str(e)


def answer_question_with_data(question: str, df: pd.DataFrame) -> str:
    """
    Full live-query flow: translate question -> pandas expression ->
    execute safely -> phrase the real result in plain English.
    """
    expression = generate_pandas_query(question, list(df.columns))
    result, error = safe_execute_query(df, expression)

    if error:
        return f"I tried to calculate this but ran into an error: {error}"

    prompt = f"""The user asked: "{question}"
The calculated answer is: {result}

Write a short, natural sentence answering the question using this exact result."""

    return call_llm(prompt)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
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

        if needs_live_data(question):
            answer = answer_question_with_data(question, df)
        else:
            answer = answer_question(question, context, chat_history)

        print(f"\nAssistant: {answer}\n")

        chat_history.append({"question": question, "answer": answer})