from fastapi import FastAPI

# This creates the actual application object.
# Everything we add later (routes, middleware, etc.) attaches to this "app".
app = FastAPI(title="AI Data Analyst Agent")


# This is a "route decorator". It tells FastAPI:
# "when someone sends a GET request to '/', run the function below"
@app.get("/")
def read_root():
    # FastAPI automatically converts this Python dict into a JSON response.
    return {"status": "Backend is alive", "project": "AI Data Analyst Agent"}


# A second route, just to show how multiple endpoints work.
@app.get("/health")
def health_check():
    return {"health": "ok"}