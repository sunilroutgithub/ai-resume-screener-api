from fastapi import FastAPI

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