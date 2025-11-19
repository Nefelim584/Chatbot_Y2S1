from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator

from sqlalchemy import delete, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session

from .data_models import CvText, CvVector, Entity, SessionLocal

__all__ = [
    "session_scope",
    "upsert_entity",
    "get_entity",
    "delete_entity",
    "upsert_cv_text",
    "get_cv_text",
    "upsert_cv_vector",
    "get_cv_vector",
    "delete_cv_data",
]


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional scope for database interactions.

    Usage:
        with session_scope() as session:
            session.add(...)
    """

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
    return {"id": entity.id, "name": entity.name, "tag": entity.tag}


def _cv_text_to_dict(cv_text: CvText) -> Dict[str, Any]:
    return {
        "id": cv_text.id,
        "tag": cv_text.tag,
        "skills": cv_text.skills,
        "experience": cv_text.experience,
        "education": cv_text.education,
        "location": cv_text.location,
    }


def _cv_vector_to_dict(cv_vector: CvVector) -> Dict[str, Any]:
    return {
        "id": cv_vector.id,
        "tag": cv_vector.tag,
        "skills": cv_vector.skills,
        "experience": cv_vector.experience,
        "education": cv_vector.education,
        "location": cv_vector.location,
    }


def _filtered_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed_fields = {"skills", "experience", "education", "location"}
    return {key: value for key, value in payload.items() if key in allowed_fields}


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------

def upsert_entity(name: str, tag: str) -> Dict[str, Any]:
    """
    Create a new Entity or update the name if the tag already exists.
    Returns a serialized dictionary of the entity.
    """

    with session_scope() as session:
        entity = session.execute(
            select(Entity).where(Entity.tag == tag)
        ).scalar_one_or_none()

        if entity is None:
            entity = Entity(name=name, tag=tag)
            session.add(entity)
        else:
            entity.name = name

        session.flush()
        data = _entity_to_dict(entity)

    return data


def get_entity(tag: str) -> Dict[str, Any]:
    """Fetch an entity by tag."""

    with session_scope() as session:
        entity = session.execute(
            select(Entity).where(Entity.tag == tag)
        ).scalar_one_or_none()

        if entity is None:
            raise NoResultFound(f"Entity with tag '{tag}' does not exist.")

        data = _entity_to_dict(entity)

    return data


def delete_entity(tag: str) -> None:
    """
    Delete an entity by tag. Cascading deletes will remove related CV data
    thanks to the ORM relationships configured in `data_models.py`.
    """

    with session_scope() as session:
        result = session.execute(delete(Entity).where(Entity.tag == tag))
        if result.rowcount == 0:
            raise NoResultFound(f"Entity with tag '{tag}' does not exist.")


# ---------------------------------------------------------------------------
# CV text operations
# ---------------------------------------------------------------------------

def upsert_cv_text(tag: str, **fields: Any) -> Dict[str, Any]:
    """
    Create or update the CvText entry for a tag.
    Raises if the owning Entity does not exist.
    """

    payload = _filtered_payload(fields)
    if not payload:
        raise ValueError("No valid CvText fields were provided.")

    with session_scope() as session:
        entity = session.execute(
            select(Entity).where(Entity.tag == tag)
        ).scalar_one_or_none()
        if entity is None:
            raise NoResultFound(
                f"Cannot upsert CvText: Entity with tag '{tag}' does not exist."
            )

        cv_text = session.execute(
            select(CvText).where(CvText.tag == tag)
        ).scalar_one_or_none()

        if cv_text is None:
            cv_text = CvText(tag=tag, **payload)
            session.add(cv_text)
        else:
            for key, value in payload.items():
                setattr(cv_text, key, value)

        session.flush()
        data = _cv_text_to_dict(cv_text)

    return data


def get_cv_text(tag: str) -> Dict[str, Any]:
    """Retrieve CvText by entity tag."""

    with session_scope() as session:
        cv_text = session.execute(
            select(CvText).where(CvText.tag == tag)
        ).scalar_one_or_none()

        if cv_text is None:
            raise NoResultFound(f"CvText for tag '{tag}' does not exist.")

        data = _cv_text_to_dict(cv_text)

    return data


# ---------------------------------------------------------------------------
# CV vector operations
# ---------------------------------------------------------------------------

def upsert_cv_vector(tag: str, **fields: Any) -> Dict[str, Any]:
    """
    Create or update the CvVector entry for a tag.
    Raises if the owning Entity does not exist.
    """

    payload = _filtered_payload(fields)
    if not payload:
        raise ValueError("No valid CvVector fields were provided.")

    with session_scope() as session:
        entity = session.execute(
            select(Entity).where(Entity.tag == tag)
        ).scalar_one_or_none()
        if entity is None:
            raise NoResultFound(
                f"Cannot upsert CvVector: Entity with tag '{tag}' does not exist."
            )

        cv_vector = session.execute(
            select(CvVector).where(CvVector.tag == tag)
        ).scalar_one_or_none()

        if cv_vector is None:
            cv_vector = CvVector(tag=tag, **payload)
            session.add(cv_vector)
        else:
            for key, value in payload.items():
                setattr(cv_vector, key, value)

        session.flush()
        data = _cv_vector_to_dict(cv_vector)

    return data


def get_cv_vector(tag: str) -> Dict[str, Any]:
    """Retrieve CvVector by entity tag."""

    with session_scope() as session:
        cv_vector = session.execute(
            select(CvVector).where(CvVector.tag == tag)
        ).scalar_one_or_none()

        if cv_vector is None:
            raise NoResultFound(f"CvVector for tag '{tag}' does not exist.")

        data = _cv_vector_to_dict(cv_vector)

    return data


# ---------------------------------------------------------------------------
# Combined helpers
# ---------------------------------------------------------------------------

def delete_cv_data(tag: str) -> None:
    """
    Remove the entire entity graph (Entity + CvText + CvVector) for a tag.
    """

    delete_entity(tag)


