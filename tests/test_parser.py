"""
Tests for resume parser module
"""
import os
import pytest
from app.parser import extract_text_from_pdf, extract_text_from_docx, extract_text_from_resume

def test_extract_text_from_pdf_nonexistent():
    """Test PDF extraction with non-existent file"""
    with pytest.raises(FileNotFoundError):
        extract_text_from_resume("nonexistent.pdf")

def test_extract_text_from_docx_nonexistent():
    """Test DOCX extraction with non-existent file"""
    with pytest.raises(FileNotFoundError):
        extract_text_from_resume("nonexistent.docx")

def test_extract_text_unsupported_format():
    """Test with unsupported file format"""
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_resume("resume.txt")