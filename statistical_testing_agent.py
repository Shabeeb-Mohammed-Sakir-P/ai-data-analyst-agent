import pandas as pd
from scipy import stats
from llm_client import call_llm
from utils import parse_json_response


def run_statistical_test(df: pd.DataFrame, hypothesis: dict) -> dict:
    """
    Runs the appropriate statistical test based on the hypothesis type,
    and returns the result including whether it's statistically significant.
    """
    hyp_type = hypothesis.get("type")
    columns = hypothesis.get("columns_involved", [])

    result = {
        "hypothesis": hypothesis.get("hypothesis"),
        "type": hyp_type,
        "test_used": None,
        "p_value": None,
        "significant": None,
        "note": None,
    }

    try:
        if hyp_type == "correlation" or hyp_type == "trend":
            result.update(_run_correlation_test(df, columns, hyp_type))
        elif hyp_type == "group_comparison":
            result.update(_run_group_comparison(df, columns))
        else:
            result["note"] = f"Unknown hypothesis type: {hyp_type}"
    except Exception as e:
        result["note"] = f"Test failed: {str(e)}"

    return result


def _run_correlation_test(df: pd.DataFrame, columns: list, hyp_type: str) -> dict:
    col_a, col_b = columns[0], columns[1]
    series_a = df[col_a]
    series_b = df[col_b]

    # For trend hypotheses, one column is often a date — convert it to a number
    if hyp_type == "trend":
        for col_name, series in [(col_a, series_a), (col_b, series_b)]:
            if not pd.api.types.is_numeric_dtype(series):
                converted = pd.to_datetime(series, errors="coerce")
                if converted.notna().mean() > 0.9:
                    if col_name == col_a:
                        series_a = converted.map(pd.Timestamp.toordinal)
                    else:
                        series_b = converted.map(pd.Timestamp.toordinal)

    if not (pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b)):
        return {"note": f"Cannot correlate: '{col_a}' or '{col_b}' is not numeric"}

    combined = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    correlation, p_value = stats.pearsonr(combined["a"], combined["b"])

    return {
        "test_used": "Pearson correlation",
        "correlation_coefficient": round(correlation, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def _run_group_comparison(df: pd.DataFrame, columns: list) -> dict:
    group_col, value_col = columns[0], columns[1]

    # If NEITHER column is numeric, this is really a comparison of
    # proportions between two categorical variables — needs a Chi-square test
    if not pd.api.types.is_numeric_dtype(df[group_col]) and not pd.api.types.is_numeric_dtype(df[value_col]):
        return _run_chi_square_test(df, group_col, value_col)

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        group_col, value_col = value_col, group_col

    clean_df = df[[group_col, value_col]].dropna()
    groups = [g[value_col].values for _, g in clean_df.groupby(group_col)]

    if len(groups) < 2:
        return {"note": f"Not enough distinct groups in '{group_col}' to compare"}

    if len(groups) == 2:
        statistic, p_value = stats.ttest_ind(groups[0], groups[1])
        test_name = "Independent t-test"
    else:
        statistic, p_value = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"

    return {
        "test_used": test_name,
        "statistic": round(statistic, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def _run_chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    Used when comparing two CATEGORICAL variables (e.g. region vs is_active),
    rather than a category vs a numeric value. Tests whether the two
    categories are related/dependent on each other.
    """
    clean_df = df[[col_a, col_b]].dropna()
    contingency_table = pd.crosstab(clean_df[col_a], clean_df[col_b])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "test_used": "Chi-square test of independence",
        "statistic": round(chi2, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def summarize_test_results(results: list) -> str:
    """
    Turns the statistical test results into a plain-English summary,
    clearly stating which hypotheses were supported and which weren't.
    """
    prompt = f"""You are a data analyst. Below are statistical test results
for several hypotheses. Write a clear 4-6 sentence summary explaining which
hypotheses were statistically significant (p < 0.05) and what that means in
plain English, and which were not supported by the data. Be specific and
avoid just repeating the numbers without explanation.

Results:
{results}
"""
    return call_llm(prompt)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    from cleaning_agent import apply_cleaning_action
    from hypothesis_agent import generate_hypotheses

    # Step 1: Load and clean the data first (chaining agents together)
    df = pd.read_csv("data/sample_messy_customers.csv")
    df = apply_cleaning_action(df, {"action": "remove_duplicates", "column": "all"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "region"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "is_active"})
    df = apply_cleaning_action(df, {"action": "fix_dtype", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "monthly_spend"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "region"})

    # Step 2: Generate hypotheses on the now-cleaned data
    hypotheses = generate_hypotheses(df)

    # Step 3: Run the actual statistical test for each hypothesis
    results = [run_statistical_test(df, h) for h in hypotheses]

    print("=== Statistical Test Results ===\n")
    for r in results:
        print(f"Hypothesis: {r['hypothesis']}")
        print(f"  Test: {r.get('test_used', 'N/A')}")
        print(f"  p-value: {r.get('p_value', 'N/A')}  |  Significant: {r.get('significant', 'N/A')}")
        if r.get("note"):
            print(f"  Note: {r['note']}")
        print()

    print("=== Plain-English Summary ===")
    print(summarize_test_results(results))