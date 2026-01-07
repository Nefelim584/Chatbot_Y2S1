import numpy as np
from sentence_transformers import SentenceTransformer


# ------------- CONFIG -------------
# Use the base model BEFORE training:
# MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Use your fine-tuned education model AFTER training:
# MODEL_PATH = "models/education-field-matcher-miniLM"

MODEL_PATH = "models/education-field-matcher-miniLM"  # <-- change to your actual path
# ----------------------------------


def encode_education_field(model: SentenceTransformer, field_str: str) -> np.ndarray:
    """
    Takes an education field string and returns an embedding.
    Example input: "Computer Science"
    """
    field_str = (field_str or "").strip()
    if not field_str:
        raise ValueError("No education field provided.")

    emb = model.encode(field_str, normalize_embeddings=True)
    return np.array(emb)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    # ---- Try some examples ----
    tests = [
        # Same / synonyms (should be high)
        ("Mechanical Engineering", "AI Engineering"),
        # ("Machine Learning", "ML"),
        # ("Informatique", "Computer Science"),
        #
        # # Very related (should be high-ish)
        # ("Artificial Intelligence", "Machine Learning"),
        # ("Data Engineering", "Big Data"),
        # ("Cloud Computing", "DevOps"),
        # ("Cybersecurity", "Network Engineering"),
        #
        # # Moderately related (should be mid)
        # ("Computer Science", "Mathematics"),
        # ("Quantum Computing", "Physics"),
        # ("Graphic Design", "Human-Computer Interaction"),
        #
        # # Unrelated (should be low)
        # ("Marketing", "Mechanical Engineering"),
        # ("Nursing", "Software Engineering"),
        # ("Law", "Deep Learning"),
    ]

    for a_text, b_text in tests:
        emb_a = encode_education_field(model, a_text)
        emb_b = encode_education_field(model, b_text)
        score = cosine(emb_a, emb_b)
        print(f"{a_text!r}  vs  {b_text!r}  ->  cosine: {score:.4f}")


if __name__ == "__main__":
    main()
