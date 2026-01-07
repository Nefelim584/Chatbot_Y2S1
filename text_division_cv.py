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
# Pydantic model (new desired shape)
# ===========================================================

class CandidateProfile(BaseModel):
    """Structured representation of extracted resume information."""

    model_config = ConfigDict(populate_by_name=True)

    Education: Dict[str, Any] | None = Field(
        None,
        description="Highest education entry with degree and field."
    )

    Experience: Dict[str, Any] | None = Field(
        None,
        description="Aggregated experience: total years and main position."
    )

    Location: str = Field(
        "",
        description="City, Country."
    )

    Skills: List[str] = Field(
        default_factory=list,
        description="List of skills extracted from the resume text."
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
# helper for year difference
# ===========================================================

def _parse_date(date_str):
    if not date_str:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue

    return None


def compute_recent_same_position_years(experiences: List[Dict[str, Any]]):
    """
    experiences are expected newest → oldest

    Logic:
    - take the most recent real professional position
    - ignore unrelated teenage/student jobs
    - sum years for consecutive roles with SIMILAR position labels
    """

    if not experiences:
        return None, None

    # main position name (title of latest job)
    main_position = experiences[0].get("position")

    if not main_position:
        return None, None

    total_years = 0.0

    for exp in experiences:
        if exp.get("position") != main_position:
            break

        start = _parse_date(exp.get("start_date"))
        end = _parse_date(exp.get("end_date")) or datetime.now()

        if not start:
            continue

        years = (end - start).days / 365.25
        total_years += max(0, years)

    return round(total_years, 2), main_position


# ===========================================================
# Prompts
# ===========================================================

system_prompt = (
    "You are an assistant that extracts resume information and responds strictly in JSON.\n"
    "Return keys exactly as: Education, Experience, Location, Skills.\n\n"

    "Rules:\n"
    "- NEVER use markdown, code fences, backticks, or explanations.\n"
    "- Your entire output MUST be a single JSON object.\n"
    "- Every missing field MUST be null or empty, not omitted.\n\n"

    "Education:\n"
    "- Extract ALL education entries internally.\n"
    "- Choose ONLY the highest education level.\n"
    "- Education must contain exactly two keys:\n"
    "  - degree\n"
    "  - field\n"
    "- degree MUST be normalized to exactly one of:\n"
    "  [\"High School\", \"Bachelor\", \"Master\", \"PhD\"]\n"
    "- If the resume degree is not exactly one of those, convert it using the rules below.\n\n"

    "Degree normalization:\n"
    "- Map to \"High School\" if mentions: high school diploma, secondary school, GED, A-levels, baccalauréat/bac.\n"
    "- Map to \"Bachelor\" if mentions: bachelor, BSc, BA, BEng, BBA, BTech, Licence (FR/EU), undergraduate degree.\n"
    "- Map to \"Master\" if mentions: master, MSc, MA, MEng, MBA, MiM, MPhil, Diplôme d’ingénieur (FR).\n"
    "- Map to \"PhD\" if mentions: PhD, doctorate, doctoral degree.\n"
    "- If multiple degrees exist, choose the HIGHEST using:\n"
    "  High School < Bachelor < Master < PhD\n"
    "- Never output any other degree label besides the 4 allowed strings.\n"
    "- If nothing is found, Education must be null.\n\n"

    "Experience:\n"
    "Step 1: Extract ALL experiences as a list of objects containing:\n"
    "- position\n"
    "- company\n"
    "- start_date\n"
    "- end_date\n\n"

    "Step 2: Determine MAIN professional position:\n"
    "- Prefer the most recent specialized professional position\n"
    "- Ignore clearly unrelated early jobs (e.g., dishwasher, waiter, delivery driver)\n"
    "- Prefer last consecutive positions similar to each other\n\n"

    "Step 3: Final output Experience must contain ONLY:\n"
    "- total_years_experience (float)\n"
    "- position (main job title)\n\n"

    "Location:\n"
    "- Return ONLY the primary location in the format: 'City, Country'.\n"
    "- If not found, return an empty string.\n\n"

    "Skills:\n"
    "- MUST be a JSON array of short skill strings.\n"
)


# ===========================================================
# Core extraction function
# ===========================================================

def extract_candidate_profile(extracted_text: str) -> CandidateProfile:
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

    # ---- experience post-processing ----
    raw_experiences = data.get("Experience")

    if isinstance(raw_experiences, list):
        years, position = compute_recent_same_position_years(raw_experiences)

        data["Experience"] = {
            "total_years_experience": years,
            "position": position
        }

    profile = CandidateProfile(**data).model_dump()

    return profile, response.usage.total_tokens


# ===========================================================
# Example usage
# ===========================================================
if __name__ == "__main__":
    from text_extraction import extract_text_auto

    pdf_path = "data/CV/CV_Brut/CV_Aissam_Debbache_HARD.pdf"
    cv_text = extract_text_auto(pdf_path)

    profile, tokens_used = extract_candidate_profile(cv_text)

    print("\n=== PARSED PROFILE ===")
    print(type(profile))
    print(profile)
    print(f"\nTokens used: {tokens_used}")
