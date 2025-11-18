from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

# Update this URL with real credentials/environment variables when deploying.
DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Entity(Base):
    """Represents the high-level entity for CV records."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    tag = Column(String(255), nullable=False, unique=True, index=True)

    cv_texts = relationship(
        "CvText",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    cv_vectors = relationship(
        "CvVector",
        back_populates="entity",
        cascade="all, delete-orphan",
    )


class CvText(Base):
    """Stores the text representation of a CV."""

    __tablename__ = "cv_text"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skills = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    education = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    tag = Column(
        String(255),
        ForeignKey("entities.tag", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    entity = relationship("Entity", back_populates="cv_texts")


class CvVector(Base):
    """Stores the vectorized representation of a CV."""

    __tablename__ = "cv_vector"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skills = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    education = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    tag = Column(
        String(255),
        ForeignKey("entities.tag", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    entity = relationship("Entity", back_populates="cv_vectors")


def init_db() -> None:
    """Create all tables in the database."""

    Base.metadata.create_all(bind=engine)

