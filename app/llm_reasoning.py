import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM_PROMPT = """You are an expert technical recruiter and ATS reasoning engine.
Given a job description and a candidate resume, evaluate the match.

Respond with ONLY valid JSON, no markdown, no preamble, in exactly this shape:
{
  "match_verdict": "Strong Match" | "Partial Match" | "Weak Match",
  "reasoning": "<2-3 sentence explanation of the verdict>",
  "matched_skills": ["<skill1>", "<skill2>"],
  "missing_skills": ["<skill1>", "<skill2>"]
}
"""


def get_llm_reasoning(job_description: str, resume_text: str) -> dict:
    """
    Calls Groq's LLM to generate a match verdict, reasoning, and skill gap analysis.
    Returns a dict matching the expected JSON shape.
    """
    user_prompt = f"""Job Description:
{job_description}

Candidate Resume:
{resume_text}

Evaluate the match and respond with the JSON object described in your instructions."""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    return json.loads(raw_content)