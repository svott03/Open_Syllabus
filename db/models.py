from sqlalchemy import (
    Column, Integer, Text, String, ForeignKey, Index, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import SYNC_DATABASE_URL

Base = declarative_base()


class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    syllabi = relationship("Syllabus", back_populates="topic")


class Syllabus(Base):
    __tablename__ = "syllabi"
    id = Column(Integer, primary_key=True)
    university = Column(String(20), nullable=False, index=True)
    college = Column(String(200))
    department = Column(String(200))
    course_number = Column(String(100))
    course_title = Column(String(300))
    subject = Column(String(50))
    term = Column(String(20))
    term_label = Column(String(50))
    section = Column(String(50))
    instructor = Column(String(200))
    url = Column(Text)
    local_path = Column(String(300))
    extracted_text = Column(Text)
    topic_id = Column(Integer, ForeignKey("topics.id"), index=True)
    topic = relationship("Topic", back_populates="syllabi")
    references = relationship("SyllabusReference", back_populates="syllabus")


class Title(Base):
    __tablename__ = "titles"
    id = Column(Integer, primary_key=True)
    canonical_title = Column(Text, nullable=False)
    authors = Column(Text)
    slug = Column(String(300), unique=True, index=True)
    references = relationship("SyllabusReference", back_populates="title")


class SyllabusReference(Base):
    __tablename__ = "syllabus_references"
    syllabus_id = Column(Integer, ForeignKey("syllabi.id"), primary_key=True)
    title_id = Column(Integer, ForeignKey("titles.id"), primary_key=True)
    syllabus = relationship("Syllabus", back_populates="references")
    title = relationship("Title", back_populates="references")

    __table_args__ = (
        Index("idx_ref_title", "title_id"),
        Index("idx_ref_syllabus", "syllabus_id"),
    )


def get_engine():
    return create_engine(SYNC_DATABASE_URL)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
