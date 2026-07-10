from fastapi import FastAPI, HTTPException

from app.models import ScreeningRequest, ScreeningResponse
from app.embeddings import compute_similarity
from app.llm_reasoning import get_llm_reasoning

app = FastAPI(
    title="AI Resume Screener API",
    description="Embeds job descriptions and resumes, ranks candidates by semantic similarity with LLM-based reasoning.",
    version="0.1.0"
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Resume Screener API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/screen", response_model=ScreeningResponse)
def screen_resume(request: ScreeningRequest):
    """
    Screens a resume against a job description.
    Combines local embedding similarity with LLM-based reasoning.
    """
    try:
        similarity_score = compute_similarity(request.job_description, request.resume_text)
        llm_result = get_llm_reasoning(request.job_description, request.resume_text)

        return ScreeningResponse(
            candidate_name=request.candidate_name,
            similarity_score=similarity_score,
            match_verdict=llm_result["match_verdict"],
            reasoning=llm_result["reasoning"],
            matched_skills=llm_result.get("matched_skills", []),
            missing_skills=llm_result.get("missing_skills", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")