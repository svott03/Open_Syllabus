"""SyllabusLens API routes — the 3 deliverable endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import func, and_, text

from db.models import get_session, Topic, Title, Syllabus, SyllabusReference

router = APIRouter()


class TopicOut(BaseModel):
    id: int
    name: str


class TitleSearchResult(BaseModel):
    id: int
    canonical_title: str
    authors: str | None
    slug: str | None


class RankingItem(BaseModel):
    title_id: int
    canonical_title: str
    authors: str | None
    slug: str | None
    count: int


class RankingsResponse(BaseModel):
    mode: str  # "assignment" or "co-assignment"
    pivot_title: TitleSearchResult | None
    total: int
    results: list[RankingItem]


# --- GET /api/topics ---

@router.get("/topics", response_model=list[TopicOut])
def list_topics():
    """Return all topics for filter dropdowns."""
    session = get_session()
    try:
        topics = session.query(Topic).order_by(Topic.name).all()
        return [TopicOut(id=t.id, name=t.name) for t in topics]
    finally:
        session.close()


# --- GET /api/titles/search ---

@router.get("/titles/search", response_model=list[TitleSearchResult])
def search_titles(
    q: str = Query("", description="Search query for title or author"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search titles by name or author (type-ahead)."""
    session = get_session()
    try:
        if not q.strip():
            titles = session.query(Title).limit(limit).all()
        else:
            q_like = f"%{q.strip()}%"
            titles = (
                session.query(Title)
                .filter(
                    (Title.canonical_title.ilike(q_like))
                    | (Title.authors.ilike(q_like))
                )
                .limit(limit)
                .all()
            )
        return [
            TitleSearchResult(
                id=t.id,
                canonical_title=t.canonical_title,
                authors=t.authors,
                slug=t.slug,
            )
            for t in titles
        ]
    finally:
        session.close()


# --- GET /api/rankings ---

@router.get("/rankings", response_model=RankingsResponse)
def get_rankings(
    schools: str | None = Query(
        None,
        description="Comma-separated school filter: osu,ufl,utaustin. None = all.",
    ),
    topics: str | None = Query(
        None,
        description="Comma-separated topic IDs. None = all.",
    ),
    title: int | None = Query(
        None,
        description="Title ID for co-assignment mode. None = assignment ranking.",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return descending (title, count) pairs.

    Without `title`: counts how many syllabi reference each title (assignment mode).
    With `title`: counts co-references — titles appearing alongside the pivot title.
    """
    session = get_session()
    try:
        school_list = None
        if schools:
            school_list = [s.strip().lower() for s in schools.split(",") if s.strip()]

        topic_ids = None
        if topics:
            topic_ids = [int(t.strip()) for t in topics.split(",") if t.strip()]

        pivot = None
        if title is not None:
            pivot = session.query(Title).get(title)

        if pivot:
            results, total = _co_assignment_query(
                session, pivot.id, school_list, topic_ids, limit, offset
            )
            return RankingsResponse(
                mode="co-assignment",
                pivot_title=TitleSearchResult(
                    id=pivot.id,
                    canonical_title=pivot.canonical_title,
                    authors=pivot.authors,
                    slug=pivot.slug,
                ),
                total=total,
                results=results,
            )
        else:
            results, total = _assignment_query(
                session, school_list, topic_ids, limit, offset
            )
            return RankingsResponse(
                mode="assignment",
                pivot_title=None,
                total=total,
                results=results,
            )
    finally:
        session.close()


def _build_syllabus_filter(school_list, topic_ids):
    """Build SQLAlchemy filter conditions for school and topic."""
    conditions = []
    if school_list:
        conditions.append(Syllabus.university.in_(school_list))
    if topic_ids:
        conditions.append(Syllabus.topic_id.in_(topic_ids))
    return conditions


def _assignment_query(session, school_list, topic_ids, limit, offset):
    """Count syllabi referencing each title, filtered by school/topic."""
    q = (
        session.query(
            Title.id,
            Title.canonical_title,
            Title.authors,
            Title.slug,
            func.count(SyllabusReference.syllabus_id).label("cnt"),
        )
        .join(SyllabusReference, SyllabusReference.title_id == Title.id)
        .join(Syllabus, Syllabus.id == SyllabusReference.syllabus_id)
    )

    for cond in _build_syllabus_filter(school_list, topic_ids):
        q = q.filter(cond)

    q = q.group_by(Title.id)

    total = q.count()

    results = (
        q.order_by(func.count(SyllabusReference.syllabus_id).desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        RankingItem(
            title_id=r[0],
            canonical_title=r[1],
            authors=r[2],
            slug=r[3],
            count=r[4],
        )
        for r in results
    ], total


def _co_assignment_query(session, pivot_title_id, school_list, topic_ids, limit, offset):
    """Count co-references: titles appearing in the same syllabi as the pivot."""
    sr1 = SyllabusReference.__table__.alias("sr1")
    sr2 = SyllabusReference.__table__.alias("sr2")

    q = (
        session.query(
            Title.id,
            Title.canonical_title,
            Title.authors,
            Title.slug,
            func.count(sr1.c.syllabus_id.distinct()).label("cnt"),
        )
        .select_from(sr1)
        .join(sr2, and_(
            sr1.c.syllabus_id == sr2.c.syllabus_id,
            sr1.c.title_id != sr2.c.title_id,
        ))
        .join(Title, Title.id == sr2.c.title_id)
        .join(Syllabus, Syllabus.id == sr1.c.syllabus_id)
        .filter(sr1.c.title_id == pivot_title_id)
    )

    for cond in _build_syllabus_filter(school_list, topic_ids):
        q = q.filter(cond)

    q = q.group_by(Title.id)

    total = q.count()

    results = (
        q.order_by(func.count(sr1.c.syllabus_id.distinct()).desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        RankingItem(
            title_id=r[0],
            canonical_title=r[1],
            authors=r[2],
            slug=r[3],
            count=r[4],
        )
        for r in results
    ], total
