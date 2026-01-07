import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from cv_pipeline import cv_processing_pipeline  # <- your pipeline file


# ---------- CONFIG ----------
CV_FOLDER = "data/CV/CV_Brut"  # <- change if needed
DB_DSN = "postgresql://cvuser:cvpass@localhost:5432/cvdb"
# ---------------------------


def to_vector(x):
    """
    Convert numpy embedding to a Python list of floats (pgvector accepts list/tuple).
    Keep None as None.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return x.astype(float).tolist()
    return None


def main():
    folder = Path(CV_FOLDER)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found or not a directory: {CV_FOLDER}")

    # Iterate through ALL files (not only PDFs)
    file_paths = sorted([p for p in folder.iterdir() if p.is_file()])
    if not file_paths:
        print(f"No files found in: {CV_FOLDER}")
        return

    print(f"Found {len(file_paths)} files in: {CV_FOLDER}")

    total_tokens_all = 0
    processed = 0
    failed = 0

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for file_path in file_paths:
                file_name = file_path.name
                print(f"\nProcessing: {file_name}")

                try:
                    profile, tokens_used = cv_processing_pipeline(str(file_path))
                    total_tokens_all += int(tokens_used or 0)
                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"  ERROR processing {file_name}: {e}")
                    continue

                edu = profile.get("Education") or {}
                exp = profile.get("Experience") or {}
                loc = profile.get("Location") or {}

                education_degree = edu.get("degree")
                education_field = to_vector(edu.get("field"))

                experience_years = exp.get("years")
                try:
                    experience_years = float(experience_years) if experience_years is not None else None
                except Exception:
                    experience_years = None

                experience_position = to_vector(exp.get("position"))

                location_lat = loc.get("latitude")
                location_lon = loc.get("longitude")

                skills_vec = to_vector(profile.get("Skills"))

                # UPSERT by file_name
                cur.execute(
                    """
                    INSERT INTO cv_profiles (
                        file_name,
                        education_degree, education_field,
                        experience_years, experience_position,
                        location_lat, location_lon,
                        skills
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (file_name) DO UPDATE SET
                        education_degree = EXCLUDED.education_degree,
                        education_field = EXCLUDED.education_field,
                        experience_years = EXCLUDED.experience_years,
                        experience_position = EXCLUDED.experience_position,
                        location_lat = EXCLUDED.location_lat,
                        location_lon = EXCLUDED.location_lon,
                        skills = EXCLUDED.skills
                    """,
                    (
                        file_name,
                        education_degree, education_field,
                        experience_years, experience_position,
                        location_lat, location_lon,
                        skills_vec
                    )
                )

                conn.commit()
                print(f"  inserted/updated ✅  tokens={tokens_used}")

    print("\n==============================")
    print("Ingestion finished.")
    print(f"Processed files: {processed}")
    print(f"Failed files:    {failed}")
    print(f"TOTAL tokens:    {total_tokens_all}")
    print("==============================\n")


if __name__ == "__main__":
    main()
