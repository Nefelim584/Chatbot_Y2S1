"""
DUAL MODEL MATCHER
Combines Model 1 (Role Matcher) and Model 2 (Experience Matcher)

Architecture:
1. Extract roles from experience text
2. Use Model 1 to get role similarity
3. If roles match well (score >= threshold), use Model 2 for experience
4. Combine scores intelligently

Scoring Strategy:
- If role_score < 0.5: Return role_score directly (or role_score^2)
- If role_score >= 0.5: Combine with experience_score
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple

class DualModelMatcher:
    """
    Combines two specialized models:
    - Model 1: Role matching
    - Model 2: Experience matching
    """

    def __init__(
            self,
            role_model_path: str = "models/role-matcher-mpnet",
            experience_model_path: str = "models/experience-matcher-mpnet",
            role_threshold: float = 0.5
    ):
        """
        Initialize both models

        Args:
            role_model_path: Path to trained role matcher
            experience_model_path: Path to trained experience matcher
            role_threshold: Minimum role score to consider experience (default: 0.5)
        """
        print("Loading models...")
        self.role_model = SentenceTransformer(role_model_path)
        self.experience_model = SentenceTransformer(experience_model_path)
        self.role_threshold = role_threshold
        print("✓ Both models loaded successfully")

    def extract_role(self, experience_text: str) -> str:
        """
        Extract role from experience text

        Examples:
            "Senior Software Engineer at Google, 8 years" → "Software Engineer"
            "Dish washer, 5 years" → "Dish washer"
            "Mid-level Backend Developer at Meta, 6 years" → "Backend Developer"
        """
        text_lower = experience_text.lower()

        # Remove common patterns
        patterns_to_remove = [
            r'\bat\s+\w+',          # "at Google"
            r'\d+\+?\s*years?',     # "5 years", "3+ years"
            r',.*$',                # Everything after comma
        ]

        for pattern in patterns_to_remove:
            text_lower = re.sub(pattern, '', text_lower)

        # Remove seniority levels to get base role
        seniority_levels = [
            'junior', 'mid-level', 'senior', 'staff', 'principal',
            'lead', 'distinguished', 'entry-level', 'associate', 'entry level'
        ]

        role = text_lower.strip()
        for level in seniority_levels:
            role = role.replace(level, '').strip()

        return role.strip()

    def get_role_similarity(self, cv_experience: str, job_experience: str) -> float:
        """
        Get role similarity using Model 1

        Returns: similarity score (0-1)
        """
        role1 = self.extract_role(cv_experience)
        role2 = self.extract_role(job_experience)

        # Encode roles
        emb1 = self.role_model.encode(role1, normalize_embeddings=True)
        emb2 = self.role_model.encode(role2, normalize_embeddings=True)

        # Cosine similarity
        similarity = float(np.dot(emb1, emb2))

        return similarity

    def get_experience_similarity(self, cv_experience: str, job_experience: str) -> float:
        """
        Get experience similarity using Model 2

        Returns: similarity score (0-1)
        """
        emb1 = self.experience_model.encode(cv_experience, normalize_embeddings=True)
        emb2 = self.experience_model.encode(job_experience, normalize_embeddings=True)

        similarity = float(np.dot(emb1, emb2))

        return similarity

    def match(
        self,
        cv_experience: str,
        job_experience: str
    ) -> Dict[str, float]:
        """
        Match CV experience against job requirements

        Returns:
            {
                'final_score': float,          # Overall match score
                'role_score': float,           # Role similarity (Model 1)
                'experience_score': float,     # Experience similarity (Model 2)
                'method': str,                 # Scoring method used
                'cv_role': str,                # Extracted CV role
                'job_role': str                # Extracted job role
            }
        """
        # Step 1: Get role similarity
        role_score = self.get_role_similarity(cv_experience, job_experience)

        # Extract roles for info
        cv_role = self.extract_role(cv_experience)
        job_role = self.extract_role(job_experience)

        # Step 2: Decision based on role score
        if role_score < self.role_threshold:
            # Roles don't match well - return role score directly
            final_score = role_score

            return {
                'final_score': final_score,
                'role_score': role_score,
                'experience_score': 0.0,
                'method': 'role_only',
                'cv_role': cv_role,
                'job_role': job_role
            }

        # Step 3: Roles match well - get experience similarity
        experience_score = self.get_experience_similarity(cv_experience, job_experience)

        # Step 4: Combine scores (weighted average: 70% role, 30% experience)
        final_score = (0.70 * role_score) + (0.30 * experience_score)

        return {
            'final_score': final_score,
            'role_score': role_score,
            'experience_score': experience_score,
            'method': 'combined (70% role + 30% experience)',
            'cv_role': cv_role,
            'job_role': job_role
        }


def main():
    """Demo usage"""

    print("="*70)
    print("DUAL MODEL MATCHER - DEMO")
    print("="*70)

    # Initialize matcher
    matcher = DualModelMatcher(
        role_threshold=0.5  # Min role score to check experience
    )

    print("\nTesting various scenarios...")
    print("="*70)

    # Test cases
    test_cases = [
        # Case 1: Role mismatch (should be very low)
        {
            "name": "Complete Role Mismatch",
            "cv": "Dish washer, 5 years",
            "job": "Software Engineer, 5+ years",
            "expected": "~0.05"
        },

        # Case 2: Over-qualified (should be high)
        {
            "name": "Over-qualified",
            "cv": "Senior Software Engineer at Google, 8 years",
            "job": "Mid-level Developer, 2+ years",
            "expected": "~0.95-1.0"
        },

        # Case 3: Exact match
        {
            "name": "Good Match",
            "cv": "Senior Software Engineer at Google, 7 years",
            "job": "Senior Software Engineer, 6+ years",
            "expected": "~0.95-1.0"
        },

        # Case 4: Under-qualified
        {
            "name": "Under-qualified",
            "cv": "Junior Developer at Google, 2 years",
            "job": "Senior Engineer, 5+ years",
            "expected": "~0.20-0.30"
        },

        # Case 5: Similar roles
        {
            "name": "Similar Roles",
            "cv": "Backend Developer at Meta, 6 years",
            "job": "Software Engineer, 5+ years",
            "expected": "~0.90-0.98"
        },

        # Case 6: Another role mismatch
        {
            "name": "Another Role Mismatch",
            "cv": "Chef, 10 years",
            "job": "Data Scientist, 8+ years",
            "expected": "~0.05"
        },

        # Case 7: Company tier difference
        {
            "name": "Company Tier Gap",
            "cv": "Senior Engineer at TCS, 8 years",
            "job": "Senior Engineer at Google, 7+ years",
            "expected": "~0.65-0.75"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*70}")
        print(f"CV:       {test['cv']}")
        print(f"Job:      {test['job']}")
        print(f"Expected: {test['expected']}")
        print("-"*70)

        result = matcher.match(test['cv'], test['job'])

        print(f"Results:")
        print(f"  CV Role:           {result['cv_role']}")
        print(f"  Job Role:          {result['job_role']}")
        print(f"  Role Score:        {result['role_score']:.3f}")
        print(f"  Experience Score:  {result['experience_score']:.3f}")
        print(f"  Final Score:       {result['final_score']:.3f}")
        print(f"  Method:            {result['method']}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo use in your code:")
    print("  from dual_model_matcher import DualModelMatcher")
    print("  matcher = DualModelMatcher()")
    print("  result = matcher.match(cv_exp, job_exp)")
    print("  score = result['final_score']")


if __name__ == "__main__":
    main()