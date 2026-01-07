import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim_between_lists(l_cv_skills: list[str], l_jo_skills: list[str]) -> float:
    result = []
    result_f_first = []
    for k in range(len(l_jo_skills)):
        for i in range(len(l_cv_skills)):
            dot_product = np.dot(l_cv_skills[i], l_jo_skills[k])
            norm_skillset = np.linalg.norm(l_cv_skills[i])
            norm_required = np.linalg.norm(l_jo_skills[k])
            cos_sim = dot_product / (norm_skillset * norm_required)
            print(f"Cosine similarity between '{l_cv_skills[i]}' and '{l_jo_skills[k]}': {cos_sim}")
            result_f_first.append(cos_sim)
        result.append(np.max(result_f_first))
        print(np.max(result_f_first))
        result_f_first = []
    average_l = cosine_similarity(np.mean(l_cv_skills, axis=0).reshape(1, -1),
                                  np.mean(l_jo_skills, axis=0).reshape(1, -1))
    return average_l