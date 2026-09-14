from typing import TypedDict, Optional
import gc

import pandas as pd

from langgraph.graph import StateGraph, END

from profiling_agent import analyze_dataset

from cleaning_agent import (
    propose_cleaning_actions,
    apply_cleaning_action,
)

from hypothesis_agent import generate_hypotheses

from statistical_testing_agent import run_statistical_test

from visualization_agent import (
    decide_chart_types,
    generate_chart,
)

from feature_engineering_agent import (
    analyze_for_feature_engineering,
    propose_feature_engineering_actions,
)

from report_agent import generate_report


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of rows kept in memory for the downstream
# analysis pipeline.
#
# The complete CSV is still profiled by profiling_agent.py,
# but the heavier downstream agents work on this bounded
# DataFrame to protect Render's 512 MB memory limit.
#
# Small datasets are NOT truncated.
PIPELINE_MAX_ROWS = 10000


# ============================================================
# PROGRESS TRACKING
# ============================================================

def update_progress(dataset_id: str, step: str):
    """
    Updates the database with which agent is currently running,
    so the frontend can display live progress.
    """

    from database import SessionLocal, Dataset

    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.dataset_id == dataset_id)
            .first()
        )

        if dataset:
            dataset.current_step = step
            db.commit()

    finally:
        db.close()


# ============================================================
# PIPELINE STATE
# ============================================================

class PipelineState(TypedDict):
    dataset_id: str
    filepath: str
    approved_cleaning_actions: Optional[list]

    # DataFrame used by downstream analysis agents.
    df: Optional[pd.DataFrame]

    profiling_findings: Optional[dict]
    cleaning_actions: Optional[list]
    hypotheses: Optional[list]
    test_results: Optional[list]
    chart_specs: Optional[list]
    chart_filepaths: Optional[list]
    fe_actions: Optional[list]
    report: Optional[str]


# ============================================================
# LOAD DATA FOR DOWNSTREAM ANALYSIS
# ============================================================

def load_analysis_dataframe(filepath: str) -> tuple[pd.DataFrame, int, bool]:
    """
    Loads a safe-sized DataFrame for the downstream pipeline.

    For datasets with <= PIPELINE_MAX_ROWS:
        The complete dataset is loaded.

    For larger datasets:
        Only the first PIPELINE_MAX_ROWS are loaded.

    Returns:
        df
        number of rows loaded
        whether the dataset was sampled/truncated
    """

    # First determine the number of rows without loading
    # the complete CSV into memory.

    total_rows = 0

    try:
        for chunk in pd.read_csv(
            filepath,
            chunksize=5000,
            low_memory=True,
        ):
            total_rows += len(chunk)

            # We only need to know whether the dataset exceeds
            # our downstream processing limit.
            if total_rows > PIPELINE_MAX_ROWS:
                del chunk
                gc.collect()
                break

            del chunk

    except Exception as e:
        raise RuntimeError(
            f"Unable to inspect dataset before analysis: {e}"
        )

    # --------------------------------------------------------
    # Small dataset
    # --------------------------------------------------------

    if total_rows <= PIPELINE_MAX_ROWS:

        df = pd.read_csv(
            filepath,
            low_memory=True,
        )

        return df, len(df), False

    # --------------------------------------------------------
    # Large dataset
    # --------------------------------------------------------

    print(
        f"Large dataset detected ({total_rows}+ rows). "
        f"Loading maximum {PIPELINE_MAX_ROWS} rows "
        f"for downstream analysis."
    )

    df = pd.read_csv(
        filepath,
        nrows=PIPELINE_MAX_ROWS,
        low_memory=True,
    )

    return df, len(df), True


# ============================================================
# PROFILING NODE
# ============================================================

def profiling_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "profiling",
    )

    print("Running Profiling Agent...")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # analyze_dataset() is now memory-efficient and processes
    # the complete CSV in chunks.
    #
    # We DO NOT load the entire CSV here.
    # --------------------------------------------------------

    findings = analyze_dataset(
        state["filepath"]
    )

    # --------------------------------------------------------
    # Load only a safe amount of data for downstream agents.
    # --------------------------------------------------------

    df, analysis_rows, was_sampled = load_analysis_dataframe(
        state["filepath"]
    )

    # Add information to the profiling results so the report
    # and frontend have context about large datasets.

    if was_sampled:

        findings["analysis_note"] = (
            f"The dataset contains approximately more than "
            f"{PIPELINE_MAX_ROWS:,} rows. Full-dataset profiling "
            f"was performed, while downstream analysis agents "
            f"used the first {analysis_rows:,} rows to stay "
            f"within the available memory limit."
        )

        findings["analysis_rows"] = analysis_rows
        findings["analysis_sampled"] = True

        print(
            f"Full dataset profiling completed. "
            f"Downstream analysis will use {analysis_rows} rows."
        )

    else:

        findings["analysis_note"] = (
            "The complete dataset was used for downstream analysis."
        )

        findings["analysis_rows"] = analysis_rows
        findings["analysis_sampled"] = False

        print(
            f"Dataset contains {analysis_rows} rows. "
            f"Complete dataset will be used."
        )

    gc.collect()

    return {
        "df": df,
        "profiling_findings": findings,
    }


# ============================================================
# CLEANING NODE
# ============================================================

def cleaning_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "cleaning",
    )

    print("Applying approved Cleaning actions...")

    actions = (
        state.get("approved_cleaning_actions")
        or []
    )

    df = state["df"]

    if df is None:
        raise RuntimeError(
            "No dataset available for cleaning."
        )

    for action in actions:

        try:

            df = apply_cleaning_action(
                df,
                action,
            )

        except Exception as e:

            print(
                f"Skipped action "
                f"{action.get('action')} "
                f"on {action.get('column')}: {e}"
            )

    gc.collect()

    return {
        "df": df,
        "cleaning_actions": actions,
    }


# ============================================================
# HYPOTHESIS NODE
# ============================================================

def hypothesis_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "hypothesis",
    )

    print("Running Hypothesis Agent...")

    df = state["df"]

    if df is None:
        raise RuntimeError(
            "No dataset available for hypothesis generation."
        )

    hypotheses = generate_hypotheses(
        df
    )

    gc.collect()

    return {
        "hypotheses": hypotheses,
    }


# ============================================================
# STATISTICAL TESTING NODE
# ============================================================

def statistical_testing_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "statistical_testing",
    )

    print("Running Statistical Testing Agent...")

    df = state["df"]

    hypotheses = (
        state.get("hypotheses")
        or []
    )

    if df is None:
        raise RuntimeError(
            "No dataset available for statistical testing."
        )

    results = []

    for hypothesis in hypotheses:

        try:

            result = run_statistical_test(
                df,
                hypothesis,
            )

            results.append(result)

        except Exception as e:

            print(
                f"Statistical test skipped: {e}"
            )

            results.append({
                "error": str(e),
                "hypothesis": hypothesis,
            })

    gc.collect()

    return {
        "test_results": results,
    }


# ============================================================
# VISUALIZATION NODE
# ============================================================

def visualization_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "visualization",
    )

    print("Running Visualization Agent...")

    df = state["df"]

    test_results = (
        state.get("test_results")
        or []
    )

    if df is None:
        raise RuntimeError(
            "No dataset available for visualization."
        )

    specs = decide_chart_types(
        test_results,
        list(df.columns),
    )

    filepaths = []

    for spec in specs:

        try:

            path = generate_chart(
                df,
                spec,
            )

            if path:
                filepaths.append(path)

        except Exception as e:

            print(
                f"Chart generation skipped: {e}"
            )

    gc.collect()

    return {
        "chart_specs": specs,
        "chart_filepaths": filepaths,
    }


# ============================================================
# FEATURE ENGINEERING NODE
# ============================================================

def feature_engineering_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "feature_engineering",
    )

    print("Running Feature Engineering Agent...")

    df = state["df"]

    if df is None:
        raise RuntimeError(
            "No dataset available for feature engineering."
        )

    analysis = analyze_for_feature_engineering(
        df
    )

    actions = propose_feature_engineering_actions(
        analysis
    )

    gc.collect()

    return {
        "fe_actions": actions,
    }


# ============================================================
# REPORT NODE
# ============================================================

def report_node(state: PipelineState) -> dict:

    update_progress(
        state["dataset_id"],
        "report",
    )

    print("Running Report Agent...")

    report = generate_report(
        state["profiling_findings"],
        state["cleaning_actions"],
        state["hypotheses"],
        state["test_results"],
        state["chart_specs"],
        state["fe_actions"],
    )

    gc.collect()

    return {
        "report": report,
    }


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline():

    builder = StateGraph(
        PipelineState
    )

    builder.add_node(
        "profiling",
        profiling_node,
    )

    builder.add_node(
        "cleaning",
        cleaning_node,
    )

    builder.add_node(
        "hypothesis",
        hypothesis_node,
    )

    builder.add_node(
        "statistical_testing",
        statistical_testing_node,
    )

    builder.add_node(
        "visualization",
        visualization_node,
    )

    builder.add_node(
        "feature_engineering",
        feature_engineering_node,
    )

    builder.add_node(
        "report",
        report_node,
    )

    # --------------------------------------------------------
    # Pipeline order
    # --------------------------------------------------------

    builder.set_entry_point(
        "profiling"
    )

    builder.add_edge(
        "profiling",
        "cleaning",
    )

    builder.add_edge(
        "cleaning",
        "hypothesis",
    )

    builder.add_edge(
        "hypothesis",
        "statistical_testing",
    )

    builder.add_edge(
        "statistical_testing",
        "visualization",
    )

    builder.add_edge(
        "visualization",
        "feature_engineering",
    )

    builder.add_edge(
        "feature_engineering",
        "report",
    )

    builder.add_edge(
        "report",
        END,
    )

    return builder.compile()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    pipeline = build_pipeline()

    print(
        "Starting full pipeline run...\n"
    )

    final_state = pipeline.invoke({

        "dataset_id": "test-run",

        "filepath": (
            "data/sample_messy_customers.csv"
        ),

        "approved_cleaning_actions": [
            {
                "action": "remove_duplicates",
                "column": "all",
            },
            {
                "action": "standardize_categories",
                "column": "region",
            },
            {
                "action": "standardize_categories",
                "column": "is_active",
            },
            {
                "action": "fix_dtype",
                "column": "age",
            },
            {
                "action": "impute_missing",
                "column": "age",
            },
            {
                "action": "impute_missing",
                "column": "monthly_spend",
            },
            {
                "action": "impute_missing",
                "column": "region",
            },
        ],
    })

    print(
        "\n" + "=" * 60
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nRows used for analysis: "
        f"{len(final_state['df'])}"
    )

    print(
        f"Hypotheses tested: "
        f"{len(final_state['hypotheses'])}"
    )

    print(
        f"Charts generated: "
        f"{len(final_state['chart_filepaths'])}"
    )

    print(
        f"\n--- Final Report ---\n"
    )

    print(
        final_state["report"]
    )