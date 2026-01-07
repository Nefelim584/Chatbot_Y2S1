from sentence_transformers import SentenceTransformer

from text_division_offer import extract_job_offer_profile
from location_coor import get_coordinates
from training.skills.test_skill_similarity import encode_skills
from training.skills.test_education_similarity import encode_education_field
from training.skills.test_experience_similarity import encode_position


MODEL_PATH_SKILLS = "training/skills/models/skill-matcher-miniLM"
MODEL_PATH_EDUCATION = "training/skills/models/education-field-matcher-miniLM"
MODEL_PATH_EXPERIENCE = "training/skills/models/position-matcher-miniLM"


def offer_processing_pipeline(offer_text):
    total_tokens = 0
    final_profile = {
        "Education": {"degree": None, "field": None},
        "Experience": {"years": None, "position": None},
        "Location": {"latitude": None, "longitude": None},
        "Skills": None
    }

    # Step 1: Parse offer to extract structured data
    profile, tokens_used = extract_job_offer_profile(offer_text)
    total_tokens += tokens_used

    # -----------------------------
    # Step 2: Geocode location
    # -----------------------------
    location_text = (profile.get("Location") or "").strip()
    if location_text:
        location_data = get_coordinates(location_text)
        if location_data:
            final_profile["Location"]["latitude"] = location_data.get("lat")
            final_profile["Location"]["longitude"] = location_data.get("lon")

    # -----------------------------
    # Step 3: Skills embedding
    # -----------------------------
    skills_list = profile.get("Skills") or []
    if isinstance(skills_list, list) and len(skills_list) > 0:
        model_skills = SentenceTransformer(MODEL_PATH_SKILLS)
        final_profile["Skills"] = encode_skills(model_skills, ", ".join(skills_list))
    else:
        final_profile["Skills"] = None

    # -----------------------------
    # Step 4: Education embedding
    # -----------------------------
    edu = profile.get("Education") or None
    if isinstance(edu, dict):
        edu_degree = (edu.get("degree") or "").strip()
        edu_field = (edu.get("field") or "").strip()

        final_profile["Education"]["degree"] = edu_degree if edu_degree else None

        if edu_field:
            model_edu = SentenceTransformer(MODEL_PATH_EDUCATION)
            final_profile["Education"]["field"] = encode_education_field(model_edu, edu_field)
        else:
            final_profile["Education"]["field"] = None
    else:
        final_profile["Education"]["degree"] = None
        final_profile["Education"]["field"] = None

    # -----------------------------
    # Step 5: Experience embedding
    # -----------------------------
    exp = profile.get("Experience") or None
    if isinstance(exp, dict):
        pos_text = (exp.get("position") or "").strip()
        years_val = exp.get("total_years_experience")

        try:
            years_val = float(years_val) if years_val is not None else 0.0
        except Exception:
            years_val = 0.0

        # For job offer: if years not mentioned, we want 0 (your requirement)
        final_profile["Experience"]["years"] = years_val

        if pos_text:
            model_exp = SentenceTransformer(MODEL_PATH_EXPERIENCE)
            final_profile["Experience"]["position"] = encode_position(model_exp, pos_text)
        else:
            # For job offer: if position missing, keep None (or you can force "" if you prefer)
            final_profile["Experience"]["position"] = None
    else:
        final_profile["Experience"]["years"] = 0.0
        final_profile["Experience"]["position"] = None

    return final_profile, total_tokens


if __name__ == "__main__":
    offer_text = """
    About the job
    ...
    """
    profile, tokens = offer_processing_pipeline(offer_text)
    print(profile)
    print(f"Total tokens used: {tokens}")
