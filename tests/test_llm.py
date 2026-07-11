"""
Tests for Groq LLM integration
"""
import os
import pytest
from app.llm import ResumeScreener

def test_llm_initialization():
    """Test that the screener initializes correctly"""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY environment variable not set")
    
    screener = ResumeScreener()
    assert screener.model is not None
    assert screener.model == "llama-3.3-70b-versatile"
    assert screener.client is not None

def test_screen_resume():
    """Test screening a resume against a job description"""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY environment variable not set")
    
    screener = ResumeScreener()
    
    resume = """
    Senior Python Developer with 7 years of experience in backend development.
    Expertise in FastAPI, Django, and microservices architecture.
    Led a team of 10 developers building a financial trading platform.
    Strong knowledge of machine learning and data pipelines.
    """
    
    job_description = """
    Looking for a Senior Python Developer with:
    - 5+ years of Python experience
    - Experience with FastAPI or Django
    - Strong understanding of microservices
    - Knowledge of financial systems is a plus
    """
    
    result = screener.screen_resume(resume, job_description)
    
    assert "evaluation" in result
    assert "score" in result
    assert 0 <= result["score"] <= 100
    assert result["status"] in ["shortlisted", "not_shortlisted"]
    assert result["model_used"] == "llama-3.3-70b-versatile"
    
    # Print the evaluation for inspection
    print("\n" + "="*50)
    print("EVALUATION RESULT:")
    print("="*50)
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")
    print(f"Model: {result['model_used']}")
    print("\nFull Evaluation:")
    print(result['evaluation'])
