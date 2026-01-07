from sentence_transformers import SentenceTransformer

from text_extraction import extract_text_auto
from text_division_cv import extract_candidate_profile
from location_coor import get_coordinates
from training.skills.test_skill_similarity import encode_skills
from training.skills.test_education_similarity import encode_education_field
from training.skills.test_experience_similarity import encode_position


MODEL_PATH_SKILLS = "training/skills/models/skill-matcher-miniLM"
MODEL_PATH_EDUCATION = "training/skills/models/education-field-matcher-miniLM"
MODEL_PATH_EXPERIENCE = "training/skills/models/position-matcher-miniLM"


def cv_processing_pipeline(cv_path):
    total_tokens = 0
    final_profile = {
        "Education": {"degree": None, "field": None},
        "Experience": {"years": None, "position": None},
        "Location": {"latitude": None, "longitude": None},
        "Skills": None
    }

    # Step 1: Extract text from CV
    cv_text = extract_text_auto(cv_path)

    # Step 2: Parse CV to extract structured data
    profile, tokens_used = extract_candidate_profile(cv_text)
    total_tokens += tokens_used

    # -----------------------------
    # Step 3: Geocode location
    # -----------------------------
    location_text = (profile.get("Location") or "").strip()
    if location_text:
        location_data = get_coordinates(location_text)
        if location_data:
            final_profile["Location"]["latitude"] = location_data.get("lat")
            final_profile["Location"]["longitude"] = location_data.get("lon")

    # -----------------------------
    # Step 4: Skills embedding
    # -----------------------------
    skills_list = profile.get("Skills") or []
    if isinstance(skills_list, list) and len(skills_list) > 0:
        model_skills = SentenceTransformer(MODEL_PATH_SKILLS)
        final_profile["Skills"] = encode_skills(model_skills, ", ".join(skills_list))
    else:
        final_profile["Skills"] = None

    # -----------------------------
    # Step 5: Education embedding
    # -----------------------------
    edu = profile.get("Education") or None
    if isinstance(edu, dict):
        edu_degree = (edu.get("degree") or "").strip()
        edu_field = (edu.get("field") or "").strip()

        # store degree (or None)
        final_profile["Education"]["degree"] = edu_degree if edu_degree else None

        # embed field only if present
        if edu_field:
            model_edu = SentenceTransformer(MODEL_PATH_EDUCATION)
            final_profile["Education"]["field"] = encode_education_field(model_edu, edu_field)
        else:
            final_profile["Education"]["field"] = None
    else:
        final_profile["Education"]["degree"] = None
        final_profile["Education"]["field"] = None

    # -----------------------------
    # Step 6: Experience embedding
    # -----------------------------
    exp = profile.get("Experience") or None
    if isinstance(exp, dict):
        pos_text = (exp.get("position") or "").strip()
        years_val = exp.get("total_years_experience")

        # years: keep numeric if possible, else None
        try:
            years_val = float(years_val) if years_val is not None else None
        except Exception:
            years_val = None
        final_profile["Experience"]["years"] = years_val

        # embed position only if present
        if pos_text:
            model_exp = SentenceTransformer(MODEL_PATH_EXPERIENCE)
            final_profile["Experience"]["position"] = encode_position(model_exp, pos_text)
        else:
            final_profile["Experience"]["position"] = None
    else:
        final_profile["Experience"]["years"] = None
        final_profile["Experience"]["position"] = None

    return final_profile, total_tokens


if __name__ == "__main__":
    cv_path = "data/CV/CV_Brut/CV_Aissam_Debbache_HARD.pdf"
    profile, tokens = cv_processing_pipeline(cv_path)
    print("\nFinal CV Profile:")
    print(profile)
    print(f"\nTotal tokens used: {tokens}")
