import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.ingestion.tasks import process_document_task
from src.ingestion.celery_app import celery_app

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, PPTX, TXT) for asynchronous vectorization.
    Returns a task ID for progress polling.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    try:
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Dispatch to Celery
    task = process_document_task.delay(str(file_path), file.filename)
    
    return {
        "task_id": task.id,
        "status": "accepted",
        "message": "Document queued for processing"
    }

@router.get("/{task_id}/status")
def get_task_status(task_id: str):
    """
    Poll the current status of an ingestion task.
    """
    res = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": res.status,
    }
    
    if res.state == "SUCCESS":
        response["result"] = res.result
    elif res.state == "FAILURE":
        response["error"] = str(res.info)
    elif res.state == "PROGRESS":
        response["progress"] = res.info
        
    return response
