import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
import asyncio
from typing import AsyncGenerator
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import shutil
import json
from pydantic import BaseModel

from config import settings
from src.rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Powered Q&A System",
    description="A local RAG system for question answering using company documents",
    version="1.0.0"
)

# Set up templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health endpoint with basic info about the API."""
    return {
        "name": "RAG-Powered Q&A System",
        "version": "1.0.0",
        "status": "running",
        "model": settings.LLM_MODEL,
        "documents_dir": str(settings.DOCUMENTS_DIR)
    }

class IngestResponse(BaseModel):
    status: str
    message: str
    documents_processed: int = 0
    error: Optional[str] = None

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Ingest documents from the documents directory.
    
    Scans the documents directory and processes all supported files,
    adding them to the vector store for retrieval.
    
    Returns:
        IngestResponse: Status and details about the ingestion process
    """
    try:
        # Check if documents directory exists and has files
        if not settings.DOCUMENTS_DIR.exists() or not any(settings.DOCUMENTS_DIR.iterdir()):
            return IngestResponse(
                status="error",
                message="No documents found to ingest",
                error="Documents directory is empty or does not exist"
            )
        
        # Call the RAG system to ingest documents
        result = rag_system.ingest_documents()
        
        # If the RAG system returns a dictionary, use it directly
        if isinstance(result, dict):
            return IngestResponse(**result)
            
        # Otherwise, create a response from the result
        return IngestResponse(
            status="success",
            message="Documents ingested successfully",
            documents_processed=1 if result else 0
        )
        
    except Exception as e:
        return IngestResponse(
            status="error",
            message="Error ingesting documents",
            error=str(e)
        )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the documents directory.
    
    Args:
        file: The file to upload. Supported formats: .txt, .pdf, .docx
        
    Returns:
        dict: Status and message indicating success or failure
    """
    # Validate file type
    allowed_extensions = {'.txt', '.pdf', '.docx'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Ensure the documents directory exists
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
        
        # Create a secure filename
        filename = file.filename
        file_path = settings.DOCUMENTS_DIR / filename
        
        # Check if file already exists
        if file_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' already exists"
            )
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            # Read the file in chunks to handle large files
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
        
        return {
            "status": "success",
            "message": f"File '{filename}' uploaded successfully",
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up the file if there was an error
        if 'file_path' in locals() and file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

class QueryRequest(BaseModel):
    question: str

async def stream_response(question: str) -> AsyncGenerator[str, None]:
    """Stream the response from the RAG system.
    
    Args:
        question: The question to ask the RAG system
        
    Yields:
        str: Chunks of the response as they're generated
    """
    try:
        # Stream the response from the RAG system
        for chunk in rag_system.query(question, stream=True):
            # Format as Server-Sent Event
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            # Small delay to allow the client to process the chunk
            await asyncio.sleep(0.01)
        
        # Send a done signal
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        # Send a done signal after error
        yield "data: [DONE]\n\n"

@app.post("/query")
async def query(query_data: QueryRequest):
    """Query the RAG system with a question.
    
    Args:
        query_data: The query request containing the question.
        
    Returns:
        dict: The response from the RAG system.
    """
    try:
        if not query_data.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_system.query(query_data.question, stream=False)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/stream")
async def query_stream(query_data: QueryRequest):
    """Stream the response from the RAG system.
    
    Args:
        query_data: The query request containing the question.
        
    Returns:
        StreamingResponse: A streaming response with the generated text.
    """
    if not query_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    return StreamingResponse(
        stream_response(query_data.question),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable buffering for nginx
        }
    )

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
