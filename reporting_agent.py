from llm_client import call_llm


def generate_report(
    profiling_findings: dict,
    cleaning_actions: list,
    hypotheses: list,
    test_results: list,
    chart_specs: list,
    feature_engineering_actions: list,
) -> str:
    """
    Synthesizes outputs from every prior agent into one coherent,
    narrated report. This agent does no new analysis — it only explains
    and connects what the other agents already found.
    """
    prompt = f"""You are a senior data analyst writing a final report for a
non-technical stakeholder. Using the information below, write a clear,
well-organized report with these sections:

1. **Overview** — dataset size and general data quality (2-3 sentences)
2. **Data Quality Issues Found** — summarize cleaning needs, in plain language
3. **Key Findings** — which hypotheses were statistically significant and what
   that means practically; mention non-significant ones briefly too
4. **Visualizations** — briefly describe what charts are available and what
   each shows
5. **Readiness for Modeling** — summarize what feature engineering is needed
   before this data could be used to train a machine learning model

Write in clear, confident prose. Use the section headers above exactly.
Do not just list raw numbers — explain what they mean and why they matter.

--- Profiling findings ---
{profiling_findings}

--- Proposed cleaning actions ---
{cleaning_actions}

--- Hypotheses tested ---
{hypotheses}

--- Statistical test results ---
{test_results}

--- Charts generated ---
{chart_specs}

--- Feature engineering proposals ---
{feature_engineering_actions}
"""

    return call_llm(prompt)


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    import pandas as pd
    from profiling_agent import analyze_dataset
    from cleaning_agent import apply_cleaning_action, propose_cleaning_actions
    from hypothesis_agent import generate_hypotheses
    from statistical_testing_agent import run_statistical_test
    from visualization_agent import decide_chart_types
    from feature_engineering_agent import analyze_for_feature_engineering, propose_feature_engineering_actions

    # Run the full pipeline, agent by agent, exactly as built so far
    print("Running Profiling Agent...")
    profiling_findings = analyze_dataset("data/sample_messy_customers.csv")

    print("Running Cleaning Agent...")
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

    print("Running Hypothesis Agent...")
    hypotheses = generate_hypotheses(df)

    print("Running Statistical Testing Agent...")
    test_results = [run_statistical_test(df, h) for h in hypotheses]

    print("Running Visualization Agent...")
    chart_specs = decide_chart_types(test_results, list(df.columns))

    print("Running Feature Engineering Agent...")
    fe_analysis = analyze_for_feature_engineering(df)
    fe_actions = propose_feature_engineering_actions(fe_analysis)

    print("Running Report Agent...\n")
    report = generate_report(
        profiling_findings, cleaning_actions, hypotheses,
        test_results, chart_specs, fe_actions
    )

    print("=" * 60)
    print(report)
    print("=" * 60)

    # Save the report to a file too, for easy viewing
    with open("generated_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nReport also saved to generated_report.md")