"""
Utility script to aggregate every value stored in the `skills` column of the
`CvText` table and print them as a single Python list.
"""

from sqlalchemy import select

from data.db_interaction import session_scope
from data.data_models import CvText


def fetch_all_skills() -> list[str]:
    """Return every non-empty skills entry as a flat list of strings."""

    with session_scope() as session:
        result = session.execute(select(CvText.skills)).scalars().all()

    return [skills for skills in result if skills]



skills = fetch_all_skills()
print(skills)

