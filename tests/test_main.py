from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("app.main.get_llm_reasoning")
def test_screen_resume_success(mock_llm):
    # Fake LLM response — no real Groq API call happens in this test
    mock_llm.return_value = {
        "match_verdict": "Strong Match",
        "reasoning": "Candidate has all required skills.",
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": [],
    }

    response = client.post("/screen", json={
        "job_description": "We need a Python developer with FastAPI experience.",
        "resume_text": "I have 3 years of Python and FastAPI experience.",
        "candidate_name": "Jane Doe",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["candidate_name"] == "Jane Doe"
    assert data["match_verdict"] == "Strong Match"
    assert 0.0 <= data["similarity_score"] <= 1.0
    assert "Python" in data["matched_skills"]


def test_screen_resume_validation_error():
    # Too short to pass min_length=20 validation — should fail before reaching the LLM
    response = client.post("/screen", json={
        "job_description": "short",
        "resume_text": "short",
    })

    assert response.status_code == 422