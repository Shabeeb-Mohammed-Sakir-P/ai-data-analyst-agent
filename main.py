import os
import uuid
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from pipeline import build_pipeline
from database import get_db, Dataset, User
from auth import hash_password, verify_password, create_access_token, get_current_user

app = FastAPI(title="AI Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory="generated_charts"), name="charts")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class UserCredentials(BaseModel):
    email: str
    password: str


class ApprovedActions(BaseModel):
    approved_actions: list


@app.get("/")
def read_root():
    return {"status": "Backend is alive", "project": "AI Data Analyst Agent"}


@app.get("/health")
def health_check():
    return {"health": "ok"}


@app.post("/signup")
def signup(credentials: UserCredentials, db: Session = Depends(get_db)):
    """
    Creates a new user account with a securely hashed password.
    """
    existing_user = db.query(User).filter(User.email == credentials.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        id=str(uuid.uuid4()),
        email=credentials.email,
        hashed_password=hash_password(credentials.password),
    )
    db.add(new_user)
    db.commit()

    token = create_access_token(new_user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/login")
def login(credentials: UserCredentials, db: Session = Depends(get_db)):
    """
    Verifies email/password and returns a JWT access token if correct.
    """
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Accepts a CSV file upload, saves it to disk, and creates a database
    record linked to the logged-in user.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_dataset = Dataset(
        dataset_id=dataset_id,
        filename=file.filename,
        status="uploaded",
        owner_id=current_user.id,
    )
    db.add(new_dataset)
    db.commit()

    return {"dataset_id": dataset_id, "filename": file.filename}


@app.post("/preview/{dataset_id}")
def preview_cleaning(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Runs Profiling + proposes Cleaning actions WITHOUT applying them.
    Returns the list so the user can approve/reject each one before
    the full pipeline runs.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this dataset")

    from profiling_agent import analyze_dataset
    from cleaning_agent import propose_cleaning_actions

    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    findings = analyze_dataset(filepath)
    proposed_actions = propose_cleaning_actions(findings)

    return {"profiling_findings": findings, "proposed_actions": proposed_actions}


def run_pipeline_in_background(dataset_id: str, filepath: str, approved_actions: list):
    """
    Runs the full agent pipeline and saves results to the database.
    Uses the user-approved cleaning actions instead of auto-approving.
    """
    from database import SessionLocal
    db = SessionLocal()

    try:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        dataset.status = "processing"
        db.commit()

        pipeline = build_pipeline()
        final_state = pipeline.invoke({
            "dataset_id": dataset_id,
            "filepath": filepath,
            "approved_cleaning_actions": approved_actions,
        })

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
def start_analysis(
    dataset_id: str,
    approved: ApprovedActions,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Triggers the full agent pipeline on a previously uploaded dataset,
    using only the cleaning actions the user approved.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this dataset")

    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    background_tasks.add_task(
        run_pipeline_in_background, dataset_id, filepath, approved.approved_actions
    )

    return {"dataset_id": dataset_id, "status": "processing"}


@app.get("/status/{dataset_id}")
def get_status(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Checks whether a pipeline run has finished. Only the owner can check.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this dataset")

    return {
        "dataset_id": dataset_id,
        "status": dataset.status,
        "current_step": dataset.current_step,
        "error_message": dataset.error_message,
    }


@app.get("/results/{dataset_id}")
def get_results(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the final pipeline results. Only the owner can view them.
    """
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this dataset")
    if not dataset.results_json:
        raise HTTPException(status_code=404, detail="Results not found or not ready yet")

    return json.loads(dataset.results_json)