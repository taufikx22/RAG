import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

from app.config.loader import load_config
from src.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    retrieved_chunks: int = 0
    error: Optional[str] = None

class IngestionResponse(BaseModel):
    processed_documents: int
    total_chunks: int
    errors: List[str]
    vector_store_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    rag_system_ready: bool
    vector_store_status: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    try:
        logger.info("Loading configuration...")
        config = load_config()
        
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(config)
        
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global rag_system
    
    if rag_system is None:
        return HealthResponse(
            status="error",
            rag_system_ready=False,
            vector_store_status="not_initialized"
        )
    
    try:
        # Check vector store status
        vector_store_info = rag_system.vector_store.get_collection_info()
        vector_store_status = "healthy"
    except Exception as e:
        logger.warning(f"Vector store health check failed: {str(e)}")
        vector_store_status = "unhealthy"
    
    return HealthResponse(
        status="ok",
        rag_system_ready=True,
        vector_store_status=vector_store_status
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system with a question."""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(request.question, top_k=request.top_k)
        
        if result.get("error"):
            return QueryResponse(
                question=request.question,
                answer="",
                error=result["answer"]
            )
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            context=result.get("context"),
            retrieval_metadata=result.get("retrieval_metadata"),
            generation_metadata=result.get("generation_metadata"),
            retrieved_chunks=result.get("retrieved_chunks", 0)
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Ingest documents into the RAG system."""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    allowed_extensions = {'.pdf', '.docx', '.html', '.txt'}
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported: {allowed_extensions}"
            )
    
    try:
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            # Create temp file with proper extension
            suffix = Path(file.filename).suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(await file.read())
            temp_file.close()
            temp_paths.append(Path(temp_file.name))
        
        # Process documents
        result = rag_system.ingest_documents(temp_paths)
        
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {str(e)}")
        
        return IngestionResponse(
            processed_documents=result["processed_documents"],
            total_chunks=result["total_chunks"],
            errors=result["errors"],
            vector_store_info=result.get("vector_store_info")
        )
    
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)."""
    try:
        config = load_config()
        
        # Remove sensitive information
        safe_config = config.copy()
        if "generation" in safe_config:
            generation = safe_config["generation"].copy()
            generation.pop("openai_api_key", None)
            generation.pop("google_api_key", None)
            safe_config["generation"] = generation
        
        return safe_config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load configuration")

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
