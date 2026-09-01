import os
import uuid
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pipeline import build_pipeline

app = FastAPI(title="AI Data Analyst Agent")

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tracks status in memory: {dataset_id: "processing" | "complete" | "failed"}
job_status = {}


@app.get("/")
def read_root():
    return {"status": "Backend is alive", "project": "AI Data Analyst Agent"}


@app.get("/health")
def health_check():
    return {"health": "ok"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, saves it to disk with a unique ID,
    and returns that ID so it can be referenced in later requests.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"dataset_id": dataset_id, "filename": file.filename}


def run_pipeline_in_background(dataset_id: str, filepath: str):
    """
    Runs the full agent pipeline. This function executes in the background,
    separate from the HTTP request/response cycle, so the client doesn't
    have to wait for it to finish.
    """
    job_status[dataset_id] = "processing"

    try:
        pipeline = build_pipeline()
        final_state = pipeline.invoke({"filepath": filepath})

        # We can't save a pandas DataFrame directly as JSON, so we exclude it
        # and save everything else the frontend will actually need
        result_to_save = {
            "profiling_findings": final_state.get("profiling_findings"),
            "cleaning_actions": final_state.get("cleaning_actions"),
            "hypotheses": final_state.get("hypotheses"),
            "test_results": final_state.get("test_results"),
            "chart_specs": final_state.get("chart_specs"),
            "chart_filepaths": final_state.get("chart_filepaths"),
            "fe_actions": final_state.get("fe_actions"),
            "report": final_state.get("report"),
        }

        result_path = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_to_save, f, indent=2, default=str)

        job_status[dataset_id] = "complete"

    except Exception as e:
        print(f"Pipeline failed for {dataset_id}: {e}")
        job_status[dataset_id] = "failed"


@app.post("/analyze/{dataset_id}")
def start_analysis(dataset_id: str, background_tasks: BackgroundTasks):
    """
    Triggers the full agent pipeline on a previously uploaded dataset.
    Returns immediately; the actual analysis runs in the background.
    """
    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset not found")

    background_tasks.add_task(run_pipeline_in_background, dataset_id, filepath)

    return {"dataset_id": dataset_id, "status": "processing"}


@app.get("/status/{dataset_id}")
def get_status(dataset_id: str):
    """
    Lets the client check whether a pipeline run has finished yet.
    """
    status = job_status.get(dataset_id, "not_found")
    return {"dataset_id": dataset_id, "status": status}


@app.get("/results/{dataset_id}")
def get_results(dataset_id: str):
    """
    Returns the final pipeline results once processing is complete.
    """
    result_path = os.path.join(RESULTS_DIR, f"{dataset_id}.json")

    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Results not found or not ready yet")

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)