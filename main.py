import math

def location_score_km(d, d50=50):
    # score=0.5 at d=d50
    if d < 0: d = 0
    return math.exp(-math.log(2) * d / d50)

