import numpy as np
import math

def location_score_km(d, d50=50):
    if d < 0:
        d = 0
    return math.exp(-math.log(2) * d / d50)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _has_emb(x) -> bool:
    return isinstance(x, np.ndarray) and x.size > 0

def degree_score(required, candidate):
    DEG_LEVEL = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    if required is None:
        return 1.0
    r = DEG_LEVEL.get(required, 0)
    c = DEG_LEVEL.get(candidate, 0)
    if c >= r:
        return 1.0
    gap = r - c
    return {1: 0.6, 2: 0.3, 3: 0.1}.get(gap, 0.0)

def years_score(years_required, years_cv):
    if years_required <= 0:
        return 1.0
    if years_cv >= years_required:
        return 1.0
    ratio = years_cv / years_required
    return ratio ** 0.5

def haversine_km(lat1, lon1, lat2, lon2):
    # Great-circle distance in km
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def score_calculation(cv_profile, offer_profile):
    score = 0.0

    # -----------------------------
    # Skills matching (embeddings)
    # -----------------------------
    cv_sk = cv_profile.get("Skills")
    off_sk = offer_profile.get("Skills")
    if _has_emb(cv_sk) and _has_emb(off_sk):
        skills_score = cosine(cv_sk, off_sk)
        skills_score = max(0.0, min(1.0, skills_score))
        score += 0.3 * skills_score

    # -----------------------------
    # Education matching
    # -----------------------------
    edu_cv = cv_profile.get("Education") or {}
    edu_off = offer_profile.get("Education") or {}

    deg_cv = edu_cv.get("degree")
    deg_off = edu_off.get("degree")
    field_cv = edu_cv.get("field")
    field_off = edu_off.get("field")

    # If offer doesn't require education (degree is None), you can treat as full match (1.0)
    if deg_off is None:
        score += 0.2 * 1.0
    else:
        if (deg_cv is not None) and _has_emb(field_cv) and _has_emb(field_off):
            degree_sc = degree_score(deg_off, deg_cv)
            field_sc = cosine(field_cv, field_off)
            field_sc = max(0.0, min(1.0, field_sc))
            education_score = 0.6 * field_sc + 0.4 * degree_sc
            score += 0.2 * education_score
        # else: missing info -> contribute 0

    # -----------------------------
    # Experience matching
    # -----------------------------
    exp_cv = cv_profile.get("Experience") or {}
    exp_off = offer_profile.get("Experience") or {}

    years_cv = exp_cv.get("years")
    years_off = exp_off.get("years")
    pos_cv = exp_cv.get("position")
    pos_off = exp_off.get("position")

    if (years_cv is not None) and (years_off is not None) and _has_emb(pos_cv) and _has_emb(pos_off):
        years_sc = years_score(float(years_off), float(years_cv))
        pos_sc = cosine(pos_cv, pos_off)
        pos_sc = max(0.0, min(1.0, pos_sc))
        exp_score = 0.6 * pos_sc + 0.4 * years_sc
        score += 0.4 * exp_score  # FIX: += not =
    # else: missing info -> contribute 0

    # -----------------------------
    # Location matching (lat/lon dicts)
    # -----------------------------
    cv_loc = cv_profile.get("Location") or {}
    off_loc = offer_profile.get("Location") or {}

    cv_lat, cv_lon = cv_loc.get("latitude"), cv_loc.get("longitude")
    off_lat, off_lon = off_loc.get("latitude"), off_loc.get("longitude")

    if None not in (cv_lat, cv_lon, off_lat, off_lon):
        dist_km = haversine_km(cv_lat, cv_lon, off_lat, off_lon)
        loc_score = location_score_km(dist_km)
        loc_score = max(0.0, min(1.0, loc_score))
        score += 0.1 * loc_score

    return score
