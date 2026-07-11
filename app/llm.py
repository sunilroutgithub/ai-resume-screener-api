"""
LLM integration with Groq for resume screening and evaluation
"""
import os
import re
from typing import List, Dict, Any, Optional
from groq import Groq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScreener:
    """
    Uses Groq LLM to evaluate resumes against job descriptions
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: The Groq model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Set it as an environment variable or pass it directly.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized Groq client with model: {model}")
    
    def screen_resume(self, resume_text: str, job_description: str, rubric: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Screen a single resume against a job description with configurable rubric.
        
        Args:
            resume_text: The full resume text
            job_description: The job description to match against
            rubric: Optional custom rubric for evaluation
            
        Returns:
            Dictionary with screening results
        """
        prompt = self._create_screening_prompt(resume_text, job_description, rubric)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR recruiter specializing in technical resume screening. Provide detailed, structured evaluations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=1.0
            )
            
            evaluation_text = response.choices[0].message.content
            return self._parse_evaluation(evaluation_text, resume_text, job_description)
            
        except Exception as e:
            logger.error(f"Error during Groq API call: {str(e)}")
            raise RuntimeError(f"Failed to screen resume: {str(e)}")
    
    def screen_resumes_batch(self, resumes: List[str], job_description: str, rubric: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Screen multiple resumes against a job description.
        
        Args:
            resumes: List of resume texts
            job_description: The job description to match against
            rubric: Optional custom rubric for evaluation
            
        Returns:
            List of screening results for each resume
        """
        results = []
        for i, resume in enumerate(resumes):
            logger.info(f"Screening resume {i+1}/{len(resumes)}")
            result = self.screen_resume(resume, job_description, rubric)
            results.append(result)
        return results
    
    def _create_screening_prompt(self, resume_text: str, job_description: str, rubric: Optional[Dict[str, Any]] = None) -> str:
        """
        Create the screening prompt for the LLM with configurable rubric.
        """
        # Default rubric if none provided
        if rubric is None:
            rubric = {
                "technical_skills_weight": 0.6,
                "experience_weight": 0.3,
                "soft_skills_weight": 0.1
            }
        
        rubric_text = self._format_rubric(rubric)
        
        return f"""
Please evaluate the following resume against the job description provided.

=== JOB DESCRIPTION ===
{job_description}

=== RESUME ===
{resume_text}

=== EVALUATION RUBRIC ===
{rubric_text}

=== EVALUATION TASK ===
Provide a comprehensive screening analysis with the following structure:

1. **Overall Fit Score**: Rate from 0-100 based on the rubric above.
   - Technical Skills Score: (0-100) * {rubric.get('technical_skills_weight', 0.6)}
   - Experience Score: (0-100) * {rubric.get('experience_weight', 0.3)}
   - Soft Skills Score: (0-100) * {rubric.get('soft_skills_weight', 0.1)}
   - **Total Score**: Sum of all weighted scores

2. **Strengths**: List the candidate's top 3-5 strengths relevant to this role.

3. **Weaknesses/Gaps**: List the key gaps between the resume and job requirements.

4. **Technical Skills Assessment**: 
   - Technical skills present in resume
   - Technical skills required but missing
   - Proficiency level of technical skills

5. **Experience Match**: 
   - Years of experience vs. required
   - Relevance of experience to role
   - Key projects or achievements

6. **Recommendation**: 
   - Should this candidate be shortlisted? (Yes/No)
   - Reason for recommendation
   - Suggested interview questions to ask

7. **Additional Notes**: Any other observations or concerns.

Format your response as a clear, well-structured evaluation with bullet points under each section.
"""

    def _format_rubric(self, rubric: Dict[str, Any]) -> str:
        """
        Format the rubric for the prompt.
        """
        formatted = []
        for key, value in rubric.items():
            # Convert snake_case to Title Case for display
            display_key = key.replace('_', ' ').title()
            formatted.append(f"- **{display_key}**: {value}")
        return "\n".join(formatted)
    
    def _parse_evaluation(self, evaluation_text: str, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured dictionary.
        """
        # Extract score from the evaluation text
        score = self._extract_score(evaluation_text)
        
        return {
            "evaluation": evaluation_text,
            "score": score,
            "resume_preview": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "job_description": job_description[:200] + "..." if len(job_description) > 200 else job_description,
            "status": "shortlisted" if score >= 70 else "not_shortlisted",
            "model_used": self.model
        }
    
    def _extract_score(self, evaluation_text: str) -> int:
        """
        Extract the numeric score from the evaluation text.
        """
        # Try multiple patterns
        patterns = [
            r"Overall Fit Score[:\s]*(\d+)",
            r"Overall Fit Score[:\s]*(\d+)\s*out of 100",
            r"Total Score[:\s]*(\d+)",
            r"Score[:\s]*(\d+)(?:\s*\/\s*100)?",
            r"Rate from 0-100[:\s]*(\d+)",
            r"(\d+)\s*\/\s*100",
            r"score of (\d+)",
            r"Score: (\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                return max(0, min(100, score))
        
        # Look for "92 out of 100" style
        match = re.search(r'(\d+)\s*out of 100', evaluation_text, re.IGNORECASE)
        if match:
            return max(0, min(100, int(match.group(1))))
        
        # Look for "92%" style
        match = re.search(r'(\d+)%', evaluation_text)
        if match:
            return max(0, min(100, int(match.group(1))))
        
        logger.warning("Could not extract score from evaluation, defaulting to 50")
        return 50