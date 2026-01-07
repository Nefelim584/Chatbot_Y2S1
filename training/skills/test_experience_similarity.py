import numpy as np
from sentence_transformers import SentenceTransformer

# ------------- CONFIG -------------
# Use the base model BEFORE training:
# MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Use your fine-tuned model AFTER training:
MODEL_PATH = "models/position-matcher-miniLM"  # <-- change to your real path
# ----------------------------------


def encode_position(model: SentenceTransformer, text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty position string.")
    emb = model.encode(text, normalize_embeddings=True)
    return np.array(emb)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized already, but keep it safe
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    # ---- ENTER YOUR TWO STRINGS HERE ----
    pos_1 = "Mechanical Engineer"
    pos_2 = "AI Engineer"
    # ------------------------------------

    emb_1 = encode_position(model, pos_1)
    emb_2 = encode_position(model, pos_2)

    score = cosine(emb_1, emb_2)
    print(f"\nCosine similarity between '{pos_1}' and '{pos_2}': {score:.4f}")


if __name__ == "__main__":
    main()
