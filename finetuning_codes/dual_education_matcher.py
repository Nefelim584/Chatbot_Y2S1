"""
DUAL EDUCATION MATCHER
Combines Model 1 (Degree Relevance) and Model 2 (Education Level)

Architecture:
1. Extract degree field and education level from candidate
2. Use Model 1 to check if degree is relevant for the role
3. If degree is relevant (score >= threshold), use Model 2 for level matching
4. Combine scores: 60% degree relevance + 40% education level

Handles:
- Country equivalents (BTech = BSc, Grande École = Master's)
- General degrees (BSc Science, Mathematics work for tech roles)
- Education level matching (Master's vs Bachelor's requirements)
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple

# Prestigious universities (Ivy League + Top Global)
PRESTIGIOUS_UNIVERSITIES = [
    # USA - Ivy League
    "harvard", "yale", "princeton", "columbia", "cornell", "dartmouth",
    "brown", "university of pennsylvania", "upenn",

    # USA - Top Tech/Engineering
    "mit", "stanford", "caltech", "carnegie mellon", "berkeley", "uc berkeley",
    "georgia tech", "university of michigan", "ut austin",

    # India - IITs
    "iit", "iit bombay", "iit delhi", "iit madras", "iit kanpur", "iit kharagpur",
    "iit roorkee", "iit guwahati", "iit hyderabad", "iiit",

    # UK
    "oxford", "cambridge", "imperial college", "ucl", "university college london",
    "edinburgh", "london school of economics", "lse",

    # Europe
    "eth zurich", "epfl", "technical university of munich", "tu munich",
    "delft", "tu delft", "kaist",

    # Asia-Pacific
    "national university of singapore", "nus", "nanyang", "ntu",
    "tsinghua", "peking university", "university of tokyo",

    # France
    "école polytechnique", "polytechnique", "centralesupelec",
    "école normale supérieure", "ens",
]

class DualEducationMatcher:
    """
    Combines two specialized models:
    - Model 1: Degree relevance (is degree suitable for role?)
    - Model 2: Education level (is education level sufficient?)
    """

    def __init__(
        self,
        #degree_model_path: str = "models/degree-relevance-matcher-miniLM",
        #level_model_path: str = "models/education-level-matcher-miniLM",
        degree_model_path: str = "models/degree-relevance-matcher-mpnet",
        level_model_path: str = "models/education-level-matcher-mpnet",

        degree_threshold: float = 0.5
    ):
        """
        Initialize both models

        Args:
            degree_model_path: Path to trained degree relevance matcher
            level_model_path: Path to trained education level matcher
            degree_threshold: Minimum degree relevance to check level (default: 0.5)
        """
        print("Loading education matching models...")
        self.degree_model = SentenceTransformer(degree_model_path)
        self.level_model = SentenceTransformer(level_model_path)
        self.degree_threshold = degree_threshold
        print("✓ Both education models loaded successfully")

    def has_prestigious_university(self, education_text: str) -> bool:
        """
        Check if education is from a prestigious university

        Returns: True if from Ivy League or top global university
        """
        text_lower = education_text.lower()

        for university in PRESTIGIOUS_UNIVERSITIES:
            if university in text_lower:
                return True

        return False

    def extract_degree_info(self, education_text: str) -> Tuple[str, str]:
        """
        Extract degree field and level from education text

        Examples:
            "BTech in Computer Science, 4 years" → ("Computer Science", "BTech 4 years")
            "BSc Science" → ("Science", "BSc")
            "Master's in Data Science, IIT Delhi, 2 years" → ("Data Science", "Master's 2 years")
            "Grande École Engineering, 5 years" → ("Engineering", "Grande École 5 years")

        Returns:
            (degree_field, education_level)
        """
        text_lower = education_text.lower()

        # Handle Grande École specially (it's a level, not a field)
        if "grande" in text_lower and "école" in text_lower:
            # Extract field after "Grande École"
            # Pattern: "Grande École Engineering" → field = "Engineering"
            pattern = r'grande\s+école\s+([^,\n]+)'
            match = re.search(pattern, text_lower)
            if match:
                degree_field = match.group(1).strip()
            else:
                degree_field = "Engineering"  # Default for Grande École

            education_level = education_text  # Keep full text for level
            return degree_field, education_level

        # Extract degree field (major/specialization)
        degree_field = education_text  # Default to full text

        # Common patterns for field extraction
        field_patterns = [
            r'in\s+([^,\n]+)',  # "BTech in Computer Science"
            r'of\s+([^,\n]+)',  # "Bachelor of Science"
            r'(?:btech|bsc|beng|mtech|msc|bachelor|master|phd)\s+([^,\n]+)',  # "BSc Computer Science"
        ]

        for pattern in field_patterns:
            match = re.search(pattern, text_lower)
            if match:
                field_candidate = match.group(1).strip()
                # Skip if it's just the university name
                if not any(uni in field_candidate for uni in ["university", "college", "institute", "iit", "mit"]):
                    degree_field = field_candidate
                    break

        # Extract education level (degree type + years)
        education_level = education_text  # Default to full text

        return degree_field, education_level

    def get_degree_relevance(self, candidate_degree: str, target_role: str) -> float:
        """
        Get degree relevance using Model 1

        Returns: relevance score (0-1)
        """
        # Encode degree field and target role
        emb1 = self.degree_model.encode(candidate_degree, normalize_embeddings=True)
        emb2 = self.degree_model.encode(target_role, normalize_embeddings=True)

        # Cosine similarity
        similarity = float(np.dot(emb1, emb2))

        return similarity

    def get_level_match(self, candidate_education: str, required_education: str) -> float:
        """
        Get education level match using Model 2

        Returns: match score (0-1)
        """
        emb1 = self.level_model.encode(candidate_education, normalize_embeddings=True)
        emb2 = self.level_model.encode(required_education, normalize_embeddings=True)

        similarity = float(np.dot(emb1, emb2))

        return similarity

    def match(
        self,
        candidate_education: str,
        required_education: str,
        target_role: str
    ) -> Dict[str, float]:
        """
        Match candidate education against requirements

        Args:
            candidate_education: e.g., "BTech in Computer Science, 4 years"
            required_education: e.g., "Bachelor's required"
            target_role: e.g., "Software Engineer"

        Returns:
            {
                'final_score': float,          # Overall match score
                'degree_relevance': float,     # Degree field relevance
                'level_match': float,          # Education level match
                'method': str,                 # Scoring method used
                'candidate_field': str,        # Extracted degree field
                'candidate_level': str         # Extracted education level
            }
        """
        # Step 1: Extract degree info
        degree_field, education_level = self.extract_degree_info(candidate_education)

        # Step 2: Get degree relevance
        degree_relevance = self.get_degree_relevance(degree_field, target_role)

        # Step 3: Decision based on degree relevance
        if degree_relevance < self.degree_threshold:
            # Degree not relevant enough - return degree score only
            final_score = degree_relevance

            return {
                'final_score': final_score,
                'degree_relevance': degree_relevance,
                'level_match': 0.0,
                'method': 'degree_only',
                'candidate_field': degree_field,
                'candidate_level': education_level,
                'prestigious_university': self.has_prestigious_university(candidate_education)
            }

        # Step 4: Degree is relevant - check education level
        level_match = self.get_level_match(candidate_education, required_education)

        # Step 5: Combine scores (weighted: 60% degree, 40% level)
        final_score = (0.60 * degree_relevance) + (0.40 * level_match)

        # Step 6: Apply prestigious university boost
        is_prestigious = self.has_prestigious_university(candidate_education)

        if is_prestigious and degree_relevance >= 0.80 and level_match >= 0.85:
            # Boost by 0.1 if from prestigious university and qualifications are strong
            final_score = min(1.0, final_score + 0.1)
            method = 'combined (60% degree + 40% level) + prestigious university boost (+0.1)'
        else:
            method = 'combined (60% degree + 40% level)'

        return {
            'final_score': final_score,
            'degree_relevance': degree_relevance,
            'level_match': level_match,
            'method': method,
            'candidate_field': degree_field,
            'candidate_level': education_level,
            'prestigious_university': is_prestigious
        }


def main():
    """Demo usage"""

    print("="*70)
    print("DUAL EDUCATION MATCHER - DEMO")
    print("="*70)

    # Initialize matcher
    matcher = DualEducationMatcher(degree_threshold=0.5)

    print("\nTesting various education scenarios...")
    print("="*70)

    # Test cases
    test_cases = [
        # Case 1: Perfect match
        {
            "name": "CS Degree for Software Engineer",
            "candidate": "BTech in Computer Science, 4 years",
            "required": "Bachelor's required",
            "role": "Software Engineer",
            "expected": "~0.90-0.95"
        },

        # Case 2: General degree (BSc Science) for tech role
        {
            "name": "BSc Science for Software Engineer",
            "candidate": "BSc Science, 3 years",
            "required": "Bachelor's required",
            "role": "Software Engineer",
            "expected": "~0.60-0.75"
        },

        # Case 3: Mathematics for Data Scientist
        {
            "name": "Mathematics for Data Scientist",
            "candidate": "BSc Mathematics, 3 years",
            "required": "Bachelor's required",
            "role": "Data Scientist",
            "expected": "~0.80-0.90"
        },

        # Case 4: Over-qualified (Master's for Bachelor's role)
        {
            "name": "Master's for Bachelor's Role",
            "candidate": "MTech in Computer Science, 2 years",
            "required": "Bachelor's required",
            "role": "Software Engineer",
            "expected": "~0.90-1.0"
        },

        # Case 5: Country equivalent (Grande École)
        {
            "name": "Grande École for Master's Role",
            "candidate": "Grande École Engineering, 5 years",
            "required": "Master's required",
            "role": "Software Engineer",
            "expected": "~0.85-0.95"
        },

        # Case 6: Irrelevant degree
        {
            "name": "Mechanical for Software Role",
            "candidate": "BTech Mechanical Engineering, 4 years",
            "required": "Bachelor's required",
            "role": "Software Engineer",
            "expected": "~0.10-0.25"
        },

        # Case 7: Under-qualified
        {
            "name": "Bachelor's for Master's Role",
            "candidate": "BSc Computer Science, 3 years",
            "required": "Master's required",
            "role": "Data Scientist",
            "expected": "~0.50-0.65"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*70}")
        print(f"Candidate:  {test['candidate']}")
        print(f"Required:   {test['required']}")
        print(f"Role:       {test['role']}")
        print(f"Expected:   {test['expected']}")
        print("-"*70)

        result = matcher.match(
            test['candidate'],
            test['required'],
            test['role']
        )

        print(f"Results:")
        print(f"  Degree Field:      {result['candidate_field']}")
        print(f"  Education Level:   {result['candidate_level']}")
        print(f"  Degree Relevance:  {result['degree_relevance']:.3f}")
        print(f"  Level Match:       {result['level_match']:.3f}")
        print(f"  Final Score:       {result['final_score']:.3f}")
        print(f"  Method:            {result['method']}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo use in your code:")
    print("  from dual_education_matcher import DualEducationMatcher")
    print("  matcher = DualEducationMatcher()")
    print("  result = matcher.match(candidate_edu, required_edu, role)")
    print("  score = result['final_score']")


if __name__ == "__main__":
    main()