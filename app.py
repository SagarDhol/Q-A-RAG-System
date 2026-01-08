import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import json

from config import settings
from src.rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Powered Q&A System",
    description="A local RAG system for question answering using company documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

# Ensure documents directory exists
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with basic info about the API."""
    return {
        "name": "RAG-Powered Q&A System",
        "version": "1.0.0",
        "status": "running",
        "model": settings.LLM_MODEL,
        "documents_dir": str(settings.DOCUMENTS_DIR)
    }

@app.post("/ingest")
async def ingest_documents():
    """Ingest documents from the documents directory."""
    try:
        result = rag_system.ingest_documents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the documents directory."""
    try:
        # Save the uploaded file
        file_path = settings.DOCUMENTS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "message": f"File {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query")
async def query(question: str, top_k: int = None):
    """Query the RAG system with a question."""
    try:
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_system.query(question, top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_structured")
async def query_structured(
    question: str = Form(...),
    response_format: str = Form(...),
    top_k: int = Form(None)
):
    """Query the RAG system with a question and get a structured response."""
    try:
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        try:
            format_dict = json.loads(response_format)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for response_format")
        
        result = rag_system.query_structured(
            question=question,
            response_format=format_dict,
            top_k=top_k
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_index():
    """Clear the vector store index."""
    try:
        result = rag_system.clear_index()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all documents in the documents directory."""
    try:
        documents = []
        for file_path in settings.DOCUMENTS_DIR.glob("*"):
            if file_path.is_file():
                documents.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        return {
            "status": "success",
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the FastAPI server."""
    print(f"Starting RAG system on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Documents directory: {settings.DOCUMENTS_DIR}")
    print(f"Using model: {settings.LLM_MODEL}")
    
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1
    )

if __name__ == "__main__":
    start()
