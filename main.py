import os
import uuid
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from pipeline import build_pipeline
from database import get_db, Dataset

app = FastAPI(title="AI Data Analyst Agent")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "Backend is alive", "project": "AI Data Analyst Agent"}


@app.get("/health")
def health_check():
    return {"health": "ok"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a CSV file upload, saves it to disk, and creates a database
    record to track it through its analysis lifecycle.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_dataset = Dataset(dataset_id=dataset_id, filename=file.filename, status="uploaded")
    db.add(new_dataset)
    db.commit()

    return {"dataset_id": dataset_id, "filename": file.filename}


def run_pipeline_in_background(dataset_id: str, filepath: str):
    """
    Runs the full agent pipeline and saves results to the database.
    Creates its own database session since this runs outside the normal
    request lifecycle.
    """
    from database import SessionLocal
    db = SessionLocal()

    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        dataset.status = "processing"
        db.commit()

        pipeline = build_pipeline()
        final_state = pipeline.invoke({"filepath": filepath})

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

        dataset.results_json = json.dumps(result_to_save, default=str)
        dataset.status = "complete"
        db.commit()

    except Exception as e:
        print(f"Pipeline failed for {dataset_id}: {e}")
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if dataset:
            dataset.status = "failed"
            dataset.error_message = str(e)
            db.commit()

    finally:
        db.close()


@app.post("/analyze/{dataset_id}")
def start_analysis(dataset_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Triggers the full agent pipeline on a previously uploaded dataset.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    background_tasks.add_task(run_pipeline_in_background, dataset_id, filepath)

    return {"dataset_id": dataset_id, "status": "processing"}


@app.get("/status/{dataset_id}")
def get_status(dataset_id: str, db: Session = Depends(get_db)):
    """
    Checks whether a pipeline run has finished, reading from the database
    instead of an in-memory dictionary — this survives server restarts.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "dataset_id": dataset_id,
        "status": dataset.status,
        "error_message": dataset.error_message,
    }


@app.get("/results/{dataset_id}")
def get_results(dataset_id: str, db: Session = Depends(get_db)):
    """
    Returns the final pipeline results, read from the database.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset or not dataset.results_json:
        raise HTTPException(status_code=404, detail="Results not found or not ready yet")

    return json.loads(dataset.results_json)