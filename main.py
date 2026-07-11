"""
AI Resume Screener API - ATS screener with ChromaDB, embeddings, and LLM reasoning
"""

print("✅ main.py is being imported!")

"""
AI Resume Screener API - ATS screener with ChromaDB, embeddings, and LLM reasoning
"""
...

import os
import tempfile
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

from app.parser import extract_text_from_resume
from app.vector_store import ResumeVectorStore
from app.llm import ResumeScreener

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Screener API",
    description="ATS screener: embeds JDs + resumes in ChromaDB, ranks by semantic similarity with LLM reasoning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
screener = None


class ScreeningRequest(BaseModel):
    job_description: str = Field(..., description="Job description text")
    rubric: Optional[Dict[str, Any]] = Field(
        default={
            "technical_skills_weight": 0.6,
            "experience_weight": 0.3,
            "soft_skills_weight": 0.1
        },
        description="Custom rubric for evaluation criteria"
    )
    resume_ids: Optional[List[str]] = Field(None, description="List of resume IDs to screen")
    top_k: int = Field(5, description="Number of top matches to return")

    
class ScreeningResponse(BaseModel):
    job_description: str
    results: List[Dict[str, Any]]
    top_k: int


class ResumeUploadResponse(BaseModel):
    resume_id: str
    filename: str
    text_preview: str
    message: str


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]


def get_components():
    global vector_store, screener
    if vector_store is None:
        try:
            vector_store = ResumeVectorStore(
                collection_name="resumes",
                persist_directory="./chroma_db"
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Vector store initialization failed: {str(e)}")
    
    if screener is None:
        try:
            screener = ResumeScreener()
            logger.info("Resume screener initialized")
        except Exception as e:
            logger.error(f"Failed to initialize screener: {e}")
            pass
    
    return vector_store, screener


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "AI Resume Screener API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    components = {
        "api": "healthy",
        "parser": "healthy",
        "chromadb": "healthy" if vector_store else "not_initialized",
        "groq": "healthy" if screener else "not_initialized"
    }
    return HealthResponse(
        status="healthy",
        components=components
    )


@app.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file (PDF or DOCX)")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.docx']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. Please upload PDF or DOCX"
        )
    
    # Read content first
    content = await file.read()
    
    # Use a temporary file with proper cleanup
    tmp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            tmp_file_path = tmp_file.name
        
        # Extract text from resume
        resume_text = extract_text_from_resume(tmp_file_path)
        
        if not resume_text or len(resume_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful text from resume"
            )
        
        vs, _ = get_components()
        
        metadata = {
            "filename": file.filename,
            "file_size": len(content),
            "file_type": file_extension[1:],
        }
        
        doc_id = vs.add_resume(resume_text, metadata)
        
        return ResumeUploadResponse(
            resume_id=doc_id,
            filename=file.filename,
            text_preview=resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
            message="Resume uploaded and indexed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")
    finally:
        # Clean up temporary file - add a small delay to ensure file handle is released
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                time.sleep(0.1)  # Small delay
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")


@app.post("/screen", response_model=ScreeningResponse)
async def screen_resumes(
    request: ScreeningRequest
):
    vs, llm = get_components()
    
    if vs is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please upload resumes first."
        )
    
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Groq API not configured. Please set GROQ_API_KEY environment variable."
        )
    
    try:
        logger.info(f"Searching for top {request.top_k} resumes")
        search_results = vs.search(request.job_description, n_results=request.top_k)
        
        if not search_results or not search_results.get('ids') or not search_results['ids'][0]:
            return ScreeningResponse(
                job_description=request.job_description,
                results=[],
                top_k=request.top_k
            )
        
        resume_ids = search_results['ids'][0]
        resume_texts = search_results['documents'][0] if search_results.get('documents') else []
        metadatas = search_results['metadatas'][0] if search_results.get('metadatas') else []
        distances = search_results['distances'][0] if search_results.get('distances') else []
        
        results = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                # Pass rubric to the LLM screener
                evaluation = llm.screen_resume(
                    resume_text=resume_text,
                    job_description=request.job_description,
                    rubric=request.rubric
                )
                
                similarity_score = round(100 * (1 - distances[i]), 2) if i < len(distances) else None
                
                result = {
                    "resume_id": resume_ids[i],
                    "similarity_score": similarity_score,
                    "llm_score": evaluation['score'],
                    "status": evaluation['status'],
                    "evaluation": evaluation['evaluation'],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "resume_preview": evaluation['resume_preview'],
                    "rubric_used": request.rubric
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating resume {i}: {str(e)}")
                results.append({
                    "resume_id": resume_ids[i] if i < len(resume_ids) else f"unknown_{i}",
                    "similarity_score": round(100 * (1 - distances[i]), 2) if i < len(distances) else None,
                    "llm_score": None,
                    "status": "error",
                    "evaluation": f"Error during evaluation: {str(e)}",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "resume_preview": resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
                    "rubric_used": request.rubric
                })
        
        results.sort(key=lambda x: (x.get('llm_score', 0) or 0), reverse=True)
        
        return ScreeningResponse(
            job_description=request.job_description,
            results=results,
            top_k=request.top_k
        )
        
    except Exception as e:
        logger.error(f"Error screening resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    vs, _ = get_components()
    
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        vs.delete_resume(resume_id)
        return {"message": f"Resume {resume_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Resume not found: {str(e)}")


@app.get("/resumes")
async def list_resumes():
    vs, _ = get_components()
    
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        all_data = vs.get_all_resumes()
        return {
            "count": len(all_data.get('ids', [])),
            "resumes": [
                {
                    "id": all_data['ids'][i],
                    "metadata": all_data['metadatas'][i] if all_data.get('metadatas') else {},
                    "text_preview": all_data['documents'][i][:200] + "..." if all_data.get('documents') else ""
                }
                for i in range(len(all_data.get('ids', [])))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list resumes: {str(e)}")


@app.delete("/resumes/all")
async def delete_all_resumes():
    vs, _ = get_components()
    
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        vs.clear_all()
        return {"message": "All resumes deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete resumes: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )