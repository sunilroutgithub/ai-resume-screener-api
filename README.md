# 🤖 AI Resume Screener API

> **ATS screener that embeds JDs + resumes in ChromaDB, ranks by semantic similarity with LLM reasoning, and provides configurable rubrics – all via a REST API.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.138.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-yellow)](https://www.trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED?logo=docker)](https://www.docker.com)
[![Groq](https://img.shields.io/badge/LLM-Groq-FF6B00)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📋 **Table of Contents**

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start (Local)](#-quick-start-local)
- [Run with Docker](#-run-with-docker)
- [API Endpoints](#-api-endpoints)
- [Example: Screening with Custom Rubric](#-example-screening-with-custom-rubric)
- [Live Demo](#-live-demo)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 **About the Project**

The **AI Resume Screener API** is a production-ready ATS (Applicant Tracking System) screener that:

- **Uploads resumes** (PDF/DOCX) → extracts text → generates embeddings → stores in ChromaDB.
- **Screens against job descriptions** → uses semantic similarity + LLM reasoning to evaluate candidates.
- **Configurable rubric** – dynamically weight technical skills, experience, soft skills, and more.
- **Dockerized** – deploy to any cloud (AWS, GCP, Azure, Render, Railway).
- **JSON output** – easy integration with HR tech, SaaS platforms, or custom frontends.

This project is built to be **free, scalable, and recruiter-friendly**. Perfect for startups, HR teams, or as a portfolio project.

---

## ✨ **Features**

| Feature | Description |
|---------|-------------|
| 📄 **Resume Parsing** | Supports PDF and DOCX files |
| 🔍 **Semantic Search** | ChromaDB vector search finds similar resumes |
| 🤖 **LLM Reasoning** | Groq (llama‑3.3‑70b) provides detailed evaluations |
| ⚖️ **Configurable Rubric** | Custom weights for technical skills, experience, soft skills |
| 📊 **JSON Output** | Structured responses ready for integration |
| 🐳 **Dockerized** | Easy deployment with Docker Compose |
| 📚 **Swagger UI** | Interactive API documentation at `/docs` |
| 🔐 **CORS Enabled** | Ready for frontend integration |
| 🗑️ **CRUD Operations** | List, delete, and clear all resumes |

---

## 🏗️ **Tech Stack**

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI |
| **Vector Database** | ChromaDB |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Groq (llama‑3.3‑70b‑versatile) |
| **Containerization** | Docker & Docker Compose |
| **Resume Parsing** | PyPDF2 + python-docx |
| **Testing** | Pytest |

---

## 🚀 **Quick Start (Local)**

### Prerequisites
- Python 3.11+
- Groq API Key (free from [console.groq.com](https://console.groq.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-resume-screener-api.git
cd ai-resume-screener-api

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY="your-groq-api-key"   # Linux/Mac
# or
set GROQ_API_KEY=your-groq-api-key        # Windows CMD
# or
$env:GROQ_API_KEY="your-groq-api-key"     # PowerShell

# 5. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000