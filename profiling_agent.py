import pandas as pd


def detect_dtype_issues(df: pd.DataFrame) -> list:
    """
    Finds columns that LOOK like they should be numeric, but contain
    some non-numeric values mixed in (a common real-world data problem).
    """
    issues = []
    for col in df.select_dtypes(include="object").columns:
        non_null_values = df[col].dropna()
        numeric_convertible = pd.to_numeric(non_null_values, errors="coerce")
        # If MOST values convert to numbers but a few don't, that's a red flag
        percent_numeric = numeric_convertible.notna().mean()
        if 0.5 < percent_numeric < 1.0:
            bad_values = non_null_values[numeric_convertible.isna()].unique().tolist()
            issues.append({
                "column": col,
                "percent_numeric": round(percent_numeric * 100, 1),
                "example_bad_values": bad_values[:5]
            })
    return issues


def analyze_dataset(filepath: str) -> dict:
    """
    Loads a CSV and computes objective facts about its quality.
    Returns a dictionary of findings — no LLM involved yet, just pandas.
    """
    df = pd.read_csv(filepath)

    findings = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {},
        "dtype_issues": detect_dtype_issues(df),
        "category_inconsistencies": {},
        "outliers": {},
    }

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = round((missing_count / len(df)) * 100, 1)
            findings["missing_values"][col] = f"{missing_count} missing ({missing_pct}%)"

    # Check text columns for inconsistent capitalization (e.g. "North" vs "north")
    for col in df.select_dtypes(include="object").columns:
        unique_values = df[col].dropna().unique()
        lowercase_groups = {}
        for val in unique_values:
            key = str(val).lower().strip()
            lowercase_groups.setdefault(key, []).append(val)

        inconsistent = {k: v for k, v in lowercase_groups.items() if len(v) > 1}
        if inconsistent:
            findings["category_inconsistencies"][col] = inconsistent

    # Check numeric columns for outliers using the IQR method
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            findings["outliers"][col] = int(outlier_count)

    return findings


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    results = analyze_dataset("data/sample_messy_customers.csv")
    import json
    print(json.dumps(results, indent=2, default=str))