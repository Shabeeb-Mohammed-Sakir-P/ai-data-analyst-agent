import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException

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