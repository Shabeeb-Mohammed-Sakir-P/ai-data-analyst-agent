import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Lets matplotlib save images without needing a display
import matplotlib.pyplot as plt
from llm_client import call_llm
from utils import parse_json_response

CHART_OUTPUT_DIR = "generated_charts"


def decide_chart_types(test_results: list, available_columns: list) -> list:
    """
    Looks at the statistical test results and decides what chart type
    best represents each one. Returns a list of chart specifications.
    """
    prompt = f"""You are a data visualization expert. For each statistical
result below, decide the best chart type to visualize it. Respond with
ONLY a JSON array (no other text, no code fences) where each item has:
- "hypothesis": copy the hypothesis text exactly
- "chart_type": one of "bar", "scatter", "line", "box"
- "columns": array of column names needed for this chart — MUST be exact
  matches from this list of actual dataset columns: {available_columns}
- "title": a short, clear chart title

Only include hypotheses where a chart would genuinely add value. Do NOT
invent column names — only use names from the provided list exactly as written.

Results:
{test_results}

Respond with ONLY the JSON array, starting with [ and ending with ]."""

    raw_response = call_llm(prompt)
    return parse_json_response(raw_response)


def generate_chart(df: pd.DataFrame, spec: dict) -> str:
    """
    Actually draws the chart using matplotlib, based on a chart specification
    (either from decide_chart_types, or from a custom user request).
    Saves it as a PNG and returns the filepath.
    """
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

    chart_type = spec.get("chart_type")
    columns = spec.get("columns", [])
    title = spec.get("title", "Chart")

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        if chart_type == "bar":
            group_col, value_col = columns[0], columns[1]
            grouped = df.groupby(group_col)[value_col].mean()
            grouped.plot(kind="bar", ax=ax)
            ax.set_ylabel(value_col)

        elif chart_type == "scatter":
            x_col, y_col = columns[0], columns[1]
            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif chart_type == "line":
            x_col, y_col = columns[0], columns[1]
            sorted_df = df.sort_values(x_col)
            ax.plot(sorted_df[x_col], sorted_df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45)

        elif chart_type == "box":
            group_col, value_col = columns[0], columns[1]
            df.boxplot(column=value_col, by=group_col, ax=ax)
            plt.suptitle("")  # removes pandas' auto-added extra title

        else:
            plt.close(fig)
            return None

        ax.set_title(title)
        plt.tight_layout()

        safe_filename = "".join(c if c.isalnum() else "_" for c in title)[:50]
        filepath = os.path.join(CHART_OUTPUT_DIR, f"{safe_filename}.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    except Exception as e:
        plt.close(fig)
        print(f"Failed to generate chart '{title}': {e}")
        return None


def create_custom_chart(df: pd.DataFrame, user_request: str) -> str:
    """
    Takes a free-text request from the user (e.g. 'show spend by region as
    a bar chart') and turns it into a chart spec, then generates it.
    This is the 'guided' visualization option from our design.
    """
    available_columns = list(df.columns)

    prompt = f"""A user wants a specific chart. Their request: "{user_request}"

Available columns in the dataset: {available_columns}

Respond with ONLY a JSON object (no other text, no code fences) with:
- "chart_type": one of "bar", "scatter", "line", "box"
- "columns": array of column names from the available list needed for this chart
- "title": a short, clear chart title

Respond with ONLY the JSON object, starting with {{ and ending with }}."""

    raw_response = call_llm(prompt)
    spec = parse_json_response(raw_response)
    return generate_chart(df, spec)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    from cleaning_agent import apply_cleaning_action
    from hypothesis_agent import generate_hypotheses
    from statistical_testing_agent import run_statistical_test

    # Load and clean the data
    df = pd.read_csv("data/sample_messy_customers.csv")
    df = apply_cleaning_action(df, {"action": "remove_duplicates", "column": "all"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "region"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "is_active"})
    df = apply_cleaning_action(df, {"action": "fix_dtype", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "monthly_spend"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "region"})
    df = df[df["monthly_spend"] < 5000]  # temporarily remove the extreme outlier for this test

    # Generate hypotheses and run tests
    hypotheses = generate_hypotheses(df)
    results = [run_statistical_test(df, h) for h in hypotheses]

    # Decide and generate charts automatically
    chart_specs = decide_chart_types(results, list(df.columns))
    print(f"Decided on {len(chart_specs)} charts to generate:\n")

    for spec in chart_specs:
        filepath = generate_chart(df, spec)
        if filepath:
            print(f"✓ Saved: {filepath}  ({spec.get('chart_type')} chart)")
        else:
            print(f"✗ Failed: {spec.get('title')}")

    # Test a custom user-requested chart too
    print("\n=== Custom chart request ===")
    custom_path = create_custom_chart(df, "show me average monthly spend by region as a bar chart")
    print(f"Custom chart saved: {custom_path}")