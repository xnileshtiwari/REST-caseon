# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import asyncio
from typing import Dict
import uuid
from Document_processing.document_processing import document_chunking_and_uploading_to_vectorstore

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store task statuses
tasks: Dict[str, Dict] = {}

async def process_document_task(file_path: str, task_id: str):
    """Background task for processing documents"""
    try:
        # Modify your document processing function to be async-compatible
        # If it's not possible to modify the original function, run it in an executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            document_chunking_and_uploading_to_vectorstore, 
            file_path
        )
        
        tasks[task_id] = {
            "status": "completed",
            "result": str(result)
        }
    except Exception as e:
        tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up file: {file_path}")

@app.post("/process-document/")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing"}
    
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start background processing
        background_tasks.add_task(process_document_task, file_path, task_id)
        
        return {"task_id": task_id, "status": "processing"}
        
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

# Add a cleanup task to remove old task statuses
@app.on_event("startup")
@app.on_event("shutdown")
async def cleanup_old_tasks():
    # In a production environment, you would want to periodically clean up old task statuses
    # and any temporary files that weren't properly cleaned up
    pass