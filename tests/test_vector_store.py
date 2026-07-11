"""
Tests for vector store
"""
import pytest
from app.vector_store import ResumeVectorStore

def test_vector_store_initialization():
    """Test that vector store initializes correctly"""
    store = ResumeVectorStore(collection_name="test_resumes", persist_directory="./test_chroma_db")
    assert store.collection_name == "test_resumes"
    assert store.count() == 0

def test_generate_embeddings():
    """Test embedding generation"""
    store = ResumeVectorStore(collection_name="test_resumes", persist_directory="./test_chroma_db")
    texts = ["This is a test resume", "Another test resume"]
    embeddings = store.generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0

def test_add_and_search_resume():
    """Test adding a resume and searching for it"""
    store = ResumeVectorStore(collection_name="test_resumes", persist_directory="./test_chroma_db")
    
    # Add a resume
    resume_text = "Python developer with 5 years of experience in machine learning"
    metadata = {"filename": "test_resume.txt", "skills": ["Python", "ML"]}
    doc_id = store.add_resume(resume_text, metadata)
    
    # Search for it
    results = store.search("machine learning python", n_results=1)
    
    assert len(results['ids']) == 1
    assert results['ids'][0][0] == doc_id
    assert "Python" in results['documents'][0][0]