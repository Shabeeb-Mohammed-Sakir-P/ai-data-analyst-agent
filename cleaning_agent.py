import pandas as pd
from llm_client import call_llm
from utils import parse_json_response


def propose_cleaning_actions(findings: dict) -> list:
    """
    Takes profiling findings and asks the LLM to propose specific cleaning
    actions. Returns a list of structured decisions, each with:
      - action: what to do (e.g. "remove_duplicates", "standardize_categories")
      - column: which column it applies to (or "all" for dataset-wide actions)
      - reason: why this is being proposed
      - severity: "low", "medium", or "high" (helps prioritize in the UI later)
    """
    prompt = f"""You are a data cleaning assistant. Based on the profiling
findings below, propose specific cleaning actions. Respond with ONLY a JSON
array (no other text, no markdown code fences) where each item has exactly
these fields: "action", "column", "reason", "severity" (must be "low",
"medium", or "high").

Valid action types: "remove_duplicates", "impute_missing", "standardize_categories",
"fix_dtype", "flag_outliers", "drop_column".

Findings:
{findings}

Respond with ONLY the JSON array, starting with [ and ending with ]."""

    raw_response = call_llm(prompt)
    return parse_json_response(raw_response)


def apply_cleaning_action(df: pd.DataFrame, action: dict) -> pd.DataFrame:
    """
    Applies ONE approved cleaning action to the DataFrame and returns the
    updated version. This is only ever called after a human has approved
    the action — it never runs automatically on its own.
    """
    action_type = action.get("action")
    column = action.get("column")

    if action_type == "remove_duplicates":
        return df.drop_duplicates().reset_index(drop=True)

    elif action_type == "impute_missing":
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median()
        else:
            fill_value = df[column].mode().iloc[0]
        df[column] = df[column].fillna(fill_value)
        return df

    elif action_type == "standardize_categories":
        df[column] = df[column].astype(str).str.strip().str.lower()
        return df

    elif action_type == "fix_dtype":
        df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    elif action_type == "drop_column":
        return df.drop(columns=[column])

    elif action_type == "flag_outliers":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[f"{column}_is_outlier"] = (df[column] < lower) | (df[column] > upper)
        return df

    else:
        print(f"Warning: unknown action type '{action_type}', skipping.")
        return df


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    from profiling_agent import analyze_dataset

    findings = analyze_dataset("data/sample_messy_customers.csv")
    actions = propose_cleaning_actions(findings)

    print(f"Proposed {len(actions)} cleaning actions:\n")
    for i, action in enumerate(actions, 1):
        print(f"{i}. [{action.get('severity', '?').upper()}] {action.get('action')} "
              f"on '{action.get('column')}'")
        print(f"   Reason: {action.get('reason')}\n")

    # Test applying one approved action
    print("\n=== Testing apply_cleaning_action ===")
    df = pd.read_csv("data/sample_messy_customers.csv")
    print(f"Before: {len(df)} rows, {df.duplicated().sum()} duplicates")

    dedup_action = {"action": "remove_duplicates", "column": "all"}
    df_cleaned = apply_cleaning_action(df, dedup_action)

    print(f"After:  {len(df_cleaned)} rows, {df_cleaned.duplicated().sum()} duplicates")