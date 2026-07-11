"""
Resume parsing utilities - extract text from PDF and DOCX files
"""
import os
from typing import Optional
import PyPDF2
from docx import Document

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract all text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as a single string
    """
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")

def extract_text_from_resume(file_path: str) -> str:
    """
    Extract text from resume file based on file extension.
    
    Args:
        file_path: Path to the resume file (.pdf or .docx)
        
    Returns:
        Extracted text as a single string
    """
    # Check file extension FIRST
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Validate format before checking existence
    if file_extension not in ['.pdf', '.docx']:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use .pdf or .docx")
    
    # Then check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    else:  # .docx
        return extract_text_from_docx(file_path)