import pandas as pd
from llm_client import call_llm
from utils import parse_json_response


def summarize_dataframe_for_llm(df: pd.DataFrame) -> dict:
    """
    Builds a compact summary of the dataset's structure — NOT the raw data —
    to give the LLM enough context to reason about it without needing to
    see every row.
    """
    summary = {"columns": {}}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            summary["columns"][col] = {
                "type": "numeric",
                "mean": round(df[col].mean(), 2),
                "min": round(df[col].min(), 2),
                "max": round(df[col].max(), 2),
            }
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            summary["columns"][col] = {
                "type": "categorical",
                "unique_count": df[col].nunique(),
                "top_values": top_values,
            }

    return summary


def generate_hypotheses(df: pd.DataFrame, max_hypotheses: int = 5) -> list:
    """
    Looks at the dataset's structure and proposes specific, testable
    questions worth investigating statistically.
    """
    data_summary = summarize_dataframe_for_llm(df)

    prompt = f"""You are a data analyst. Based on this dataset summary, propose
up to {max_hypotheses} specific, testable hypotheses worth investigating.
Respond with ONLY a JSON array (no other text, no code fences) where each
item has exactly these fields:
- "hypothesis": a clear, specific question (e.g. "Does monthly_spend differ
  significantly between regions?")
- "type": one of "correlation", "group_comparison", "trend"
- "columns_involved": array of column names this hypothesis needs

Dataset summary:
{data_summary}

Respond with ONLY the JSON array, starting with [ and ending with ]."""

    raw_response = call_llm(prompt)
    return parse_json_response(raw_response)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    df = pd.read_csv("data/sample_messy_customers.csv")
    hypotheses = generate_hypotheses(df)

    print(f"Generated {len(hypotheses)} hypotheses:\n")
    for i, h in enumerate(hypotheses, 1):
        print(f"{i}. [{h.get('type')}] {h.get('hypothesis')}")
        print(f"   Columns: {h.get('columns_involved')}\n")