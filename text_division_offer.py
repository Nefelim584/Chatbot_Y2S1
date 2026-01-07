from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any

from mistralai import Mistral


# ===========================================================
# Pydantic model (same shape)
# ===========================================================

class CandidateProfile(BaseModel):
    """Structured representation of extracted job-offer information."""

    model_config = ConfigDict(populate_by_name=True)

    Education: Dict[str, Any] | None = Field(
        None,
        description="Highest education requirement with normalized degree and field."
    )

    Experience: Dict[str, Any] | None = Field(
        None,
        description="Aggregated experience requirement: years and main position."
    )

    Location: str = Field(
        "",
        description="City, Country."
    )

    Skills: List[str] = Field(
        default_factory=list,
        description="List of skills extracted from the job offer text."
    )


# ===========================================================
# JSON extraction helper (unchanged)
# ===========================================================

def _extract_json_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    json_str = text[start: end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = re.sub(r",\s*}", "}", json_str)
        return json.loads(json_str)


# ===========================================================
# Mistral client
# ===========================================================

API_KEY = os.getenv("MISTRAL_API_KEY", "")
MODEL = "mistral-small-latest"

client = Mistral(api_key=API_KEY)


# ===========================================================
# helper for years parsing
# ===========================================================

def _extract_years_number(text: str) -> float:
    """
    Extracts a years-of-experience requirement from raw text.
    Returns 0.0 if not found.
    Handles patterns like:
      - "3 years", "3+ years", "at least 5 years"
      - "minimum 2 years", "2 yrs", "2 years of experience"
    """
    if not text:
        return 0.0

    t = text.lower()

    # Prefer explicit experience requirement phrases first
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\s+(?:of\s+)?experience",
        r"(?:minimum|min\.?|at\s+least)\s*(\d+(?:\.\d+)?)\s*(?:years|year|yrs|yr)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)",
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return 0.0


# ===========================================================
# Prompts (job offer version)
# ===========================================================

system_prompt = (
    "You are an assistant that extracts information from a JOB OFFER and responds strictly in JSON.\n"
    "Return keys exactly as: Education, Experience, Location, Skills.\n\n"

    "Rules:\n"
    "- NEVER use markdown, code fences, backticks, or explanations.\n"
    "- Your entire output MUST be a single JSON object.\n"
    "- Every missing field MUST be null or empty, not omitted.\n\n"

    "Education (job requirements):\n"
    "- Extract education requirements only if explicitly mentioned.\n"
    "- If education is NOT mentioned, Education must be null.\n"
    "- If education is mentioned, Education must contain exactly two keys:\n"
    "  - degree\n"
    "  - field\n"
    "- degree MUST be normalized to exactly one of:\n"
    "  [\"High School\", \"Bachelor\", \"Master\", \"PhD\"]\n"
    "- field must be the field of study required/preferred. If not explicitly stated,\n"
    "  set field to the job position domain (same domain as Experience.position).\n\n"

    "Degree normalization:\n"
    "- Map to \"High School\" if mentions: high school diploma, secondary school, GED, A-levels, baccalauréat/bac.\n"
    "- Map to \"Bachelor\" if mentions: bachelor, BSc, BA, BEng, BBA, BTech, Licence (FR/EU), undergraduate degree.\n"
    "- Map to \"Master\" if mentions: master, MSc, MA, MEng, MBA, MiM, MPhil, Diplôme d’ingénieur (FR).\n"
    "- Map to \"PhD\" if mentions: PhD, doctorate, doctoral degree.\n"
    "- If multiple degrees exist, choose the HIGHEST using:\n"
    "  High School < Bachelor < Master < PhD\n"
    "- Never output any other degree label besides the 4 allowed strings.\n\n"

    "Experience (job requirements):\n"
    "- Experience must contain ONLY:\n"
    "  - total_years_experience\n"
    "  - position\n"
    "- position: Extract the job title / role being hired for.\n"
    "- If position is not explicitly stated, infer the most likely position from the offer.\n"
    "- total_years_experience: Extract required years of experience as a float.\n"
    "- If years of experience is not mentioned, set total_years_experience to 0.\n\n"

    "Location:\n"
    "- Return ONLY the job location in the format: 'City, Country'.\n"
    "- If not found, return an empty string.\n\n"

    "Skills:\n"
    "- MUST be a JSON array of short skill strings.\n"
    "- Include ALL skills/technologies/keywords found anywhere in the job offer.\n"
)


# ===========================================================
# Core extraction function
# ===========================================================

def extract_job_offer_profile(extracted_text: str) -> CandidateProfile:
    user_prompt = extracted_text

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    model_text = response.choices[0].message.content
    data = _extract_json_object(model_text)

    # ---- enforce years rule (0 if missing / not numeric) ----
    exp = data.get("Experience")

    if isinstance(exp, dict):
        yrs = exp.get("total_years_experience")
        try:
            yrs = float(yrs)
        except Exception:
            # fallback: try parse from raw offer text
            yrs = _extract_years_number(extracted_text)

        if yrs is None:
            yrs = 0.0

        if yrs < 0:
            yrs = 0.0

        exp["total_years_experience"] = yrs

        # position fallback (if empty)
        pos = (exp.get("position") or "").strip()
        if not pos:
            # best-effort: don't break schema, leave empty string
            exp["position"] = ""

        data["Experience"] = exp

    else:
        # if model failed, force schema
        data["Experience"] = {"total_years_experience": 0.0, "position": ""}

    # ---- education rule: only if mentioned. If present, ensure field fallback to position domain ----
    edu = data.get("Education")
    if isinstance(edu, dict):
        degree = (edu.get("degree") or "").strip()
        field = (edu.get("field") or "").strip()

        # If model returned empty degree/field, treat as not mentioned
        if not degree and not field:
            data["Education"] = None
        else:
            if not field:
                # fallback to position string (domain) as requested
                position = ""
                if isinstance(data.get("Experience"), dict):
                    position = (data["Experience"].get("position") or "").strip()
                edu["field"] = position
            data["Education"] = edu
    else:
        data["Education"] = None

    profile = CandidateProfile(**data).model_dump()
    return profile, response.usage.total_tokens


# ===========================================================
# Example usage
# ===========================================================
if __name__ == "__main__":
    # Here you pass raw job offer text (or extracted text from PDF)
    job_offer_text = """
    About the job
Cognite operates at the forefront of industrial digitalization, building AI and data solutions that solve some of the world’s hardest, highest-impact problems. With unmatched industrial heritage and a comprehensive suite of AI capabilities, including low-code AI agents, Cognite accelerates the digital transformation to drive operational improvements.

Our moonshot is bold: unlock $100B in customer value by 2035 and redefine how global industry works.

What Cognite is Relentless to achieve

We thrive in challenges. We challenge assumptions. We execute with speed and ownership. If you view obstacles as signals to step forward - not step back - you’ll feel at home here. Join us in this venture where AI and data meet ingenuity, and together, we forge the path to a smarter, more connected industrial future.

How you’ll demonstrate Ownership

We're looking for a Data Engineer who is ready to tackle big challenges and advance their career. In this role, you'll join a dynamic team committed to creating cutting-edge solutions that significantly impact critical industries such as Power & Utilities, Energy, and Manufacturing. You'll collaborate with industry leaders, solution architects, data scientists and project managers, all dedicated to deploying and optimizing digital solutions that empower our clients to make informed business decisions.

Lead the design and implementation of scalable and efficient data engineering solutions using our platform Cognite Data Fusion®.
Drive and manage integrations, extractions, data modeling, and analysis using Cognite data connectors and SQL, Python/Java and Rest APIs.
Create custom data models for data discovery, mapping, and cleansing.
Collaborate with data scientists, project managers and solution architects engineers on project deliveries to enable our customers to achieve the full potential of our industrial dataops platform.
Conduct code reviews and implement best practices to ensure high-quality and maintainable code and deliveries.
Support customers and partners in conducting data engineering tasks with Cognite products.
Contribute to the development of Cognite’s official tools and SDKs.
Collaborate with our Engineering and Product Management teams to turn customer needs into a prioritized pipeline of product offerings.

The Impact you bring to Cognite

Have a DevOps mindset, and experience with Git, CI/CD, deployment environments
Enjoys working in cross-functional teams
Able to independently investigate and solve problems
Humility to ask for help and enjoy sharing knowledge with others

Required Qualifications

Minimum 3-5 years of relevant experience in a customer-facing Data intense role
Experience delivering production-grade data pipelines using e.g. Python, SQL and Rest APIs
Experience with distributed computing such as Kubernetes and managed cloud services such as GCP and/or Azure

Preferred Experience

Bachelor or Master degree in computer science or similar. Relevant experience can compensate for formal education

Equal Opportunity

Cognite is committed to creating a diverse and inclusive environment at work and is proud to be an equal opportunity employer. All qualified applicants will receive the same level of consideration for employment.


    """

    profile, tokens_used = extract_job_offer_profile(job_offer_text)

    print("\n=== PARSED JOB OFFER PROFILE ===")
    print(type(profile))
    print(profile)
    print(f"\nTokens used: {tokens_used}")
