import os
import uuid
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from supabase import create_client, Client

from pipeline import build_pipeline
from database import get_db, Dataset, User
from auth import hash_password, verify_password, create_access_token, get_current_user


app = FastAPI(title="AI Data Analyst Agent")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SUPABASE STORAGE
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required"
    )

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
)

STORAGE_BUCKET = "datasets"


# ============================================================
# LOCAL GENERATED CHARTS
# ============================================================

CHARTS_DIR = "generated_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

app.mount(
    "/charts",
    StaticFiles(directory=CHARTS_DIR),
    name="charts"
)


# ============================================================
# MODELS
# ============================================================

class UserCredentials(BaseModel):
    email: str
    password: str


class ApprovedActions(BaseModel):
    approved_actions: list


class ChatMessage(BaseModel):
    question: str
    chat_history: list = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def upload_dataset_to_storage(dataset_id: str, file: UploadFile):
    """
    Uploads the user's CSV directly to Supabase Storage.

    The file is stored using the dataset_id as its unique filename.
    """

    storage_path = f"{dataset_id}.csv"

    try:
        file.file.seek(0)

        supabase.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=file.file,
            file_options={
                "content-type": "text/csv",
                "upsert": "false",
            },
        )

        return storage_path

    except Exception as e:
        print(f"Supabase upload failed for {dataset_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to store the uploaded dataset."
        )


def download_dataset_to_tempfile(dataset_id: str) -> str:
    """
    Downloads a dataset from Supabase Storage to a temporary local file.

    The temporary file is only used while the current operation is running.
    It does NOT depend on Render's permanent filesystem.
    """

    storage_path = f"{dataset_id}.csv"

    try:
        file_bytes = (
            supabase.storage
            .from_(STORAGE_BUCKET)
            .download(storage_path)
        )

        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        )

        temp_file.write(file_bytes)
        temp_file.close()

        return temp_file.name

    except Exception as e:
        print(f"Supabase download failed for {dataset_id}: {e}")

        raise HTTPException(
            status_code=404,
            detail="The original dataset could not be retrieved from storage."
        )


def delete_temp_file(filepath: str):
    """
    Deletes a temporary downloaded dataset after use.
    """

    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Could not remove temporary file {filepath}: {e}")


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def read_root():
    return {
        "status": "Backend is alive",
        "project": "AI Data Analyst Agent"
    }


@app.get("/health")
def health_check():
    return {"health": "ok"}


# ============================================================
# AUTH
# ============================================================

@app.post("/signup")
def signup(
    credentials: UserCredentials,
    db: Session = Depends(get_db)
):
    existing_user = (
        db.query(User)
        .filter(User.email == credentials.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_user = User(
        id=str(uuid.uuid4()),
        email=credentials.email,
        hashed_password=hash_password(credentials.password),
    )

    db.add(new_user)
    db.commit()

    token = create_access_token(new_user.id)

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.post("/login")
def login(
    credentials: UserCredentials,
    db: Session = Depends(get_db)
):
    user = (
        db.query(User)
        .filter(User.email == credentials.email)
        .first()
    )

    if not user or not verify_password(
        credentials.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )

    token = create_access_token(user.id)

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    dataset_id = str(uuid.uuid4())

    # Upload CSV to persistent Supabase Storage
    upload_dataset_to_storage(
        dataset_id,
        file
    )

    # Store dataset metadata in PostgreSQL
    new_dataset = Dataset(
        dataset_id=dataset_id,
        filename=file.filename,
        status="uploaded",
        owner_id=current_user.id,
    )

    db.add(new_dataset)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "filename": file.filename
    }


# ============================================================
# PREVIEW
# ============================================================

@app.post("/preview/{dataset_id}")
def preview_cleaning(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this dataset"
        )

    from profiling_agent import analyze_dataset
    from cleaning_agent import propose_cleaning_actions

    filepath = download_dataset_to_tempfile(dataset_id)

    try:
        findings = analyze_dataset(filepath)

        proposed_actions = propose_cleaning_actions(
            findings
        )

        return {
            "profiling_findings": findings,
            "proposed_actions": proposed_actions
        }

    finally:
        delete_temp_file(filepath)


# ============================================================
# BACKGROUND PIPELINE
# ============================================================

def run_pipeline_in_background(
    dataset_id: str,
    approved_actions: list
):
    from database import SessionLocal

    db = SessionLocal()
    filepath = None

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.dataset_id == dataset_id)
            .first()
        )

        if not dataset:
            print(f"Dataset {dataset_id} not found")
            return

        dataset.status = "processing"
        db.commit()

        # Download dataset temporarily from Supabase
        filepath = download_dataset_to_tempfile(
            dataset_id
        )

        pipeline = build_pipeline()

        final_state = pipeline.invoke({
            "dataset_id": dataset_id,
            "filepath": filepath,
            "approved_cleaning_actions": approved_actions,
        })

        result_to_save = {
            "profiling_findings": final_state.get(
                "profiling_findings"
            ),
            "cleaning_actions": final_state.get(
                "cleaning_actions"
            ),
            "hypotheses": final_state.get(
                "hypotheses"
            ),
            "test_results": final_state.get(
                "test_results"
            ),
            "chart_specs": final_state.get(
                "chart_specs"
            ),
            "chart_filepaths": final_state.get(
                "chart_filepaths"
            ),
            "fe_actions": final_state.get(
                "fe_actions"
            ),
            "report": final_state.get(
                "report"
            ),
        }

        dataset.results_json = json.dumps(
            result_to_save,
            default=str
        )

        dataset.status = "complete"

        db.commit()

    except Exception as e:

        print(
            f"Pipeline failed for {dataset_id}: {e}"
        )

        dataset = (
            db.query(Dataset)
            .filter(Dataset.dataset_id == dataset_id)
            .first()
        )

        if dataset:
            dataset.status = "failed"
            dataset.error_message = str(e)
            db.commit()

    finally:

        delete_temp_file(filepath)

        db.close()


# ============================================================
# START ANALYSIS
# ============================================================

@app.post("/analyze/{dataset_id}")
def start_analysis(
    dataset_id: str,
    approved: ApprovedActions,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this dataset"
        )

    background_tasks.add_task(
        run_pipeline_in_background,
        dataset_id,
        approved.approved_actions
    )

    return {
        "dataset_id": dataset_id,
        "status": "processing"
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/status/{dataset_id}")
def get_status(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this dataset"
        )

    return {
        "dataset_id": dataset_id,
        "status": dataset.status,
        "current_step": dataset.current_step,
        "error_message": dataset.error_message,
    }


# ============================================================
# RESULTS
# ============================================================

@app.get("/results/{dataset_id}")
def get_results(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this dataset"
        )

    if not dataset.results_json:
        raise HTTPException(
            status_code=404,
            detail="Results not found or not ready yet"
        )

    return json.loads(dataset.results_json)


# ============================================================
# CHAT
# ============================================================

@app.post("/chat/{dataset_id}")
def chat_with_results(
    dataset_id: str,
    message: ChatMessage,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Answers a question about the analysis.

    General questions use the saved analysis results.

    Questions requiring live data download the original CSV
    temporarily from Supabase Storage.
    """

    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this dataset"
        )

    if not dataset.results_json:
        raise HTTPException(
            status_code=404,
            detail="Results not ready yet"
        )

    import pandas as pd

    from qa_agent import (
        build_context_summary,
        answer_question,
        needs_live_data,
        answer_question_with_data,
    )

    results = json.loads(
        dataset.results_json
    )

    context = build_context_summary(
        results.get("profiling_findings"),
        results.get("hypotheses"),
        results.get("test_results"),
        results.get("report"),
    )

    if needs_live_data(message.question):

        filepath = None

        try:
            # Retrieve the persistent dataset
            filepath = download_dataset_to_tempfile(
                dataset_id
            )

            df = pd.read_csv(filepath)

            answer = answer_question_with_data(
                message.question,
                df
            )

        except HTTPException:
            raise

        except Exception as e:
            print(
                f"Live data question failed for {dataset_id}: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="Unable to analyze the original dataset."
            )

        finally:
            delete_temp_file(filepath)

    else:

        answer = answer_question(
            message.question,
            context,
            message.chat_history
        )

    return {
        "answer": answer
    }