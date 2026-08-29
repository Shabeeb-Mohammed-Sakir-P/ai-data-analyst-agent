import json
import re
from llm_client import call_llm


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
    return _parse_json_response(raw_response)


def _parse_json_response(raw_response: str) -> list:
    """
    LLMs sometimes wrap JSON in ```json code fences or add extra text
    despite instructions not to. This function cleans that up before parsing.
    """
    cleaned = raw_response.strip()

    # Remove markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Warning: Could not parse LLM response as JSON. Raw response was:")
        print(raw_response)
        return []


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