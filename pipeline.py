from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END

from profiling_agent import analyze_dataset
from cleaning_agent import propose_cleaning_actions, apply_cleaning_action
from hypothesis_agent import generate_hypotheses
from statistical_testing_agent import run_statistical_test
from visualization_agent import decide_chart_types, generate_chart
from feature_engineering_agent import analyze_for_feature_engineering, propose_feature_engineering_actions
from report_agent import generate_report


def update_progress(dataset_id: str, step: str):
    """
    Updates the database with which agent is currently running,
    so the frontend can display live progress.
    """
    from database import SessionLocal, Dataset
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if dataset:
            dataset.current_step = step
            db.commit()
    finally:
        db.close()


class PipelineState(TypedDict):
    dataset_id: str
    filepath: str
    df: Optional[pd.DataFrame]
    profiling_findings: Optional[dict]
    cleaning_actions: Optional[list]
    hypotheses: Optional[list]
    test_results: Optional[list]
    chart_specs: Optional[list]
    chart_filepaths: Optional[list]
    fe_actions: Optional[list]
    report: Optional[str]


def profiling_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "profiling")
    print("Running Profiling Agent...")
    df = pd.read_csv(state["filepath"])
    findings = analyze_dataset(state["filepath"])
    return {"df": df, "profiling_findings": findings}


def cleaning_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "cleaning")
    print("Running Cleaning Agent...")
    actions = propose_cleaning_actions(state["profiling_findings"])

    df = state["df"]
    for action in actions:
        try:
            df = apply_cleaning_action(df, action)
        except Exception as e:
            print(f"  Skipped action {action.get('action')} on {action.get('column')}: {e}")

    return {"df": df, "cleaning_actions": actions}


def hypothesis_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "hypothesis")
    print("Running Hypothesis Agent...")
    hypotheses = generate_hypotheses(state["df"])
    return {"hypotheses": hypotheses}


def statistical_testing_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "statistical_testing")
    print("Running Statistical Testing Agent...")
    results = [run_statistical_test(state["df"], h) for h in state["hypotheses"]]
    return {"test_results": results}


def visualization_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "visualization")
    print("Running Visualization Agent...")
    df = state["df"]
    specs = decide_chart_types(state["test_results"], list(df.columns))

    filepaths = []
    for spec in specs:
        path = generate_chart(df, spec)
        if path:
            filepaths.append(path)

    return {"chart_specs": specs, "chart_filepaths": filepaths}


def feature_engineering_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "feature_engineering")
    print("Running Feature Engineering Agent...")
    analysis = analyze_for_feature_engineering(state["df"])
    actions = propose_feature_engineering_actions(analysis)
    return {"fe_actions": actions}


def report_node(state: PipelineState) -> dict:
    update_progress(state["dataset_id"], "report")
    print("Running Report Agent...")
    report = generate_report(
        state["profiling_findings"],
        state["cleaning_actions"],
        state["hypotheses"],
        state["test_results"],
        state["chart_specs"],
        state["fe_actions"],
    )
    return {"report": report}


def build_pipeline():
    builder = StateGraph(PipelineState)

    builder.add_node("profiling", profiling_node)
    builder.add_node("cleaning", cleaning_node)
    builder.add_node("hypothesis", hypothesis_node)
    builder.add_node("statistical_testing", statistical_testing_node)
    builder.add_node("visualization", visualization_node)
    builder.add_node("feature_engineering", feature_engineering_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("profiling")
    builder.add_edge("profiling", "cleaning")
    builder.add_edge("cleaning", "hypothesis")
    builder.add_edge("hypothesis", "statistical_testing")
    builder.add_edge("statistical_testing", "visualization")
    builder.add_edge("visualization", "feature_engineering")
    builder.add_edge("feature_engineering", "report")
    builder.add_edge("report", END)

    return builder.compile()


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    pipeline = build_pipeline()

    print("Starting full pipeline run...\n")
    final_state = pipeline.invoke({"dataset_id": "test-run", "filepath": "data/sample_messy_customers.csv"})

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nRows after cleaning: {len(final_state['df'])}")
    print(f"Hypotheses tested: {len(final_state['hypotheses'])}")
    print(f"Charts generated: {len(final_state['chart_filepaths'])}")
    print(f"\n--- Final Report ---\n")
    print(final_state["report"])