import numpy as np
from sentence_transformers import SentenceTransformer


# ------------- CONFIG -------------
# Use the base model BEFORE training:
# MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Use your fine-tuned model AFTER training:
# MODEL_PATH = "models/skill-matcher-miniLM"
MODEL_PATH = "/Users/ivmakiv/Documents/Study Aivancity/project AI clinic/AIclinicproject2/AIclinic PGE2 1 sem/training/skills/models/skill-matcher-miniLM"
# ----------------------------------

# MODEL_PATH = "intfloat/e5-small-v2"

# MODEL_PATH = "models/skill-matcher-e5-small"


def encode_skills(model: SentenceTransformer, skills_str: str) -> np.ndarray:
    """
    Takes a comma-separated string of skills and returns an embedding.
    Example input: "python, django, docker"
    """
    # Split by comma, strip spaces, drop empties
    skills = [s.strip() for s in skills_str.split(",") if s.strip()]
    if not skills:
        raise ValueError("No skills provided.")

    text = "; ".join(skills)
    emb = model.encode(text, normalize_embeddings=True)
    # ensure it's a numpy array
    return np.array(emb)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)


    # cv_input = "scikit-learn, pandas, numpy, pytorch"
    # job_input = "machine learning"
    # # 0,2342

    cv_input = "public speaking"
    job_input = "electrical engineering"
    # # 0.0655

    # cv_input = "python"
    # job_input = "python"
    # 0,4146

    # cv_input = "cleaning the floor, washing dishes, cooking"
    # job_input = "python, sql, data analysis"

    emb_cv = encode_skills(model, cv_input)
    emb_job = encode_skills(model, job_input)
    print(emb_cv)

    score = cosine(emb_cv, emb_job)
    print("cosine similarity between public speaking and electrical engineering:")
    print(f"\nCosine similarity: {score:.4f}")

if __name__ == "__main__":
    main()
