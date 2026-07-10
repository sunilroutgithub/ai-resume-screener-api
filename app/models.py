from pydantic import BaseModel, Field
from typing import Optional


class ScreeningRequest(BaseModel):
    """Input: a job description and a candidate resume to compare."""
    job_description: str = Field(..., min_length=20, description="Full text of the job description")
    resume_text: str = Field(..., min_length=20, description="Full text of the candidate's resume")
    candidate_name: Optional[str] = Field(None, description="Optional candidate name for reference")


class ScreeningResponse(BaseModel):
    """Output: similarity score plus LLM reasoning for the match."""
    candidate_name: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score between 0 and 1")
    match_verdict: str = Field(..., description="LLM-generated verdict, e.g. 'Strong Match', 'Partial Match', 'Weak Match'")
    reasoning: str = Field(..., description="LLM-generated explanation of the score")
    matched_skills: list[str] = Field(default_factory=list, description="Skills found in both JD and resume")
    missing_skills: list[str] = Field(default_factory=list, description="Skills in JD but not found in resume")