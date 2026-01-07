import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from offer_pipeline import offer_processing_pipeline
from comparision import score_calculation


# ---------- CONFIG ----------
DB_DSN = "postgresql://cvuser:cvpass@localhost:5432/cvdb"
TABLE = "cv_profiles"
TOP_K = 10  # default top K
# ---------------------------


def _to_np(vec):
    """
    DB returns pgvector as list/tuple-like (after register_vector).
    Convert to numpy array. Keep None as None.
    """
    if vec is None:
        return None
    return np.array(vec, dtype=float)


def load_all_cvs_from_db(conn):
    """
    Returns list of cv_profile dicts in the SAME STRUCTURE
    expected by score_calculation().
    """
    query = f"""
        SELECT
            file_name,
            education_degree,
            education_field,
            experience_years,
            experience_position,
            location_lat,
            location_lon,
            skills
        FROM {TABLE}
    """

    cvs = []
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    for (
        file_name,
        edu_degree,
        edu_field_vec,
        exp_years,
        exp_pos_vec,
        lat,
        lon,
        skills_vec,
    ) in rows:
        cv_profile = {
            "file_name": file_name,
            "Education": {
                "degree": edu_degree,
                "field": _to_np(edu_field_vec),
            },
            "Experience": {
                "years": float(exp_years) if exp_years is not None else None,
                "position": _to_np(exp_pos_vec),
            },
            "Location": {
                "latitude": lat,
                "longitude": lon,
            },
            "Skills": _to_np(skills_vec),
        }
        cvs.append(cv_profile)

    return cvs


def rank_cvs_for_offer_text(offer_text: str, top_k: int = TOP_K):
    """
    Main reusable function for web app:
    - takes offer text
    - runs offer pipeline
    - scores all CVs from DB
    - returns top_k sorted results
    """
    offer_text = (offer_text or "").strip()
    if not offer_text:
        raise ValueError("Empty job offer text.")

    # Run offer pipeline
    offer_profile, offer_tokens = offer_processing_pipeline(offer_text)

    results = []

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)

        cvs = load_all_cvs_from_db(conn)
        if not cvs:
            return {
                "offer_tokens": int(offer_tokens or 0),
                "total_cvs": 0,
                "top_k": top_k,
                "results": []
            }

        for cv in cvs:
            s = float(score_calculation(cv, offer_profile))
            results.append((cv["file_name"], s))

    # Sort descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Slice top_k (if top_k is None, return all)
    to_print = results if top_k is None else results[:top_k]

    payload = {
        "offer_tokens": int(offer_tokens or 0),
        "total_cvs": len(results),
        "top_k": top_k,
        "results": [
            {"rank": i + 1, "file_name": fn, "score": float(sc)}
            for i, (fn, sc) in enumerate(to_print)
        ],
    }
    return payload


def main():
    offer_text = """
    About the job
    ...
    """

    data = rank_cvs_for_offer_text(offer_text, top_k=TOP_K)

    print(f"\nOffer processed using {data['offer_tokens']} tokens.\n")
    print(f"Total CVs scored: {data['total_cvs']}")
    print("Top results:\n")

    for r in data["results"]:
        print(f"{r['rank']:>3}. {r['file_name']}  |  score={r['score']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
