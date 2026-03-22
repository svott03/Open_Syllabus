"""Run the full SyllabusLens data pipeline.

Usage:
    python pipeline/run_pipeline.py --sample    # Process sample PDFs only
    python pipeline/run_pipeline.py --full      # Process all PDFs from GCP (not yet implemented)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CSV_PATH, SAMPLES_DIR, DATA_DIR
from db.models import (
    init_db, get_session, Syllabus, Topic, Title, SyllabusReference
)
from pipeline.text_extraction import extract_text
from pipeline.topic_mapping import map_subject_to_topic
from pipeline.citation_extraction import (
    extract_citations_from_text, deduplicate_titles
)


def load_csv_rows(sample_only=True):
    """Load syllabus metadata from CSV, optionally filtering to sample files."""
    sample_files = set()
    if sample_only:
        for school_dir in SAMPLES_DIR.iterdir():
            if school_dir.is_dir():
                for pdf in school_dir.glob("*.pdf"):
                    sample_files.add(pdf.name)

    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if not row["local_path"].strip():
                continue
            if sample_only:
                fname = Path(row["local_path"]).name
                if fname not in sample_files:
                    continue
            rows.append(row)
    return rows


def step_1_load_metadata(session, rows):
    """Load syllabus metadata into the database."""
    print(f"\n=== Step 1: Loading {len(rows)} syllabus metadata rows ===")
    existing = session.query(Syllabus.local_path).all()
    existing_paths = {r[0] for r in existing}

    loaded = 0
    for row in rows:
        if row["local_path"] in existing_paths:
            continue
        syl = Syllabus(
            university=row["university"],
            college=row.get("college", ""),
            department=row.get("department", ""),
            course_number=row.get("course_number", ""),
            course_title=row.get("course_title", ""),
            subject=row.get("subject", ""),
            term=row.get("term", ""),
            term_label=row.get("term_label", ""),
            section=row.get("section", ""),
            instructor=row.get("instructor", ""),
            url=row.get("url", ""),
            local_path=row["local_path"],
        )
        session.add(syl)
        loaded += 1

    session.commit()
    print(f"  Loaded {loaded} new rows (skipped {len(rows) - loaded} existing)")


def step_2_assign_topics(session):
    """Create topics and assign them to syllabi."""
    print(f"\n=== Step 2: Assigning topics ===")
    syllabi = session.query(Syllabus).filter(Syllabus.topic_id.is_(None)).all()
    print(f"  {len(syllabi)} syllabi need topic assignment")

    topic_cache = {}
    for t in session.query(Topic).all():
        topic_cache[t.name] = t

    assigned = 0
    for syl in syllabi:
        topic_name = map_subject_to_topic(syl.subject, syl.department)
        if not topic_name:
            continue

        if topic_name not in topic_cache:
            topic = Topic(name=topic_name)
            session.add(topic)
            session.flush()
            topic_cache[topic_name] = topic

        syl.topic_id = topic_cache[topic_name].id
        assigned += 1

    session.commit()
    total_topics = session.query(Topic).count()
    print(f"  Assigned topics to {assigned} syllabi ({total_topics} unique topics)")


def step_3_extract_text(session, sample_only=True):
    """Extract text from PDFs."""
    print(f"\n=== Step 3: Extracting text from PDFs ===")
    syllabi = session.query(Syllabus).filter(
        Syllabus.extracted_text.is_(None),
        Syllabus.local_path.isnot(None),
    ).all()
    print(f"  {len(syllabi)} syllabi need text extraction")

    success = 0
    failed = 0
    for i, syl in enumerate(syllabi):
        if sample_only:
            pdf_path = SAMPLES_DIR / syl.local_path
        else:
            pdf_path = DATA_DIR / "pdfs" / syl.local_path

        if not pdf_path.exists():
            alt = SAMPLES_DIR / syl.university / Path(syl.local_path).name
            if alt.exists():
                pdf_path = alt
            else:
                continue

        text, method = extract_text(str(pdf_path))
        if text:
            syl.extracted_text = text
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            session.commit()
            print(f"  Progress: {i+1}/{len(syllabi)} ({success} ok, {failed} failed)")

    session.commit()
    print(f"  Extracted text: {success} success, {failed} failed")


def step_4_extract_citations(session):
    """Extract citations from syllabus text and create title records."""
    print(f"\n=== Step 4: Extracting citations ===")
    syllabi = session.query(Syllabus).filter(
        Syllabus.extracted_text.isnot(None),
    ).all()

    already_processed = set(
        r[0] for r in session.query(SyllabusReference.syllabus_id).distinct().all()
    )
    to_process = [s for s in syllabi if s.id not in already_processed]
    print(f"  {len(to_process)} syllabi to process ({len(already_processed)} already done)")

    all_citations = []
    for syl in to_process:
        cits = extract_citations_from_text(syl.extracted_text)
        for c in cits:
            all_citations.append((syl.id, c))

    print(f"  Raw citations extracted: {len(all_citations)}")

    if not all_citations:
        print("  No citations found, skipping deduplication")
        return

    prev_citations = []
    for ref in session.query(SyllabusReference).all():
        title = session.query(Title).get(ref.title_id)
        if title:
            from pipeline.citation_extraction import RawCitation
            prev_citations.append((ref.syllabus_id, RawCitation(
                title=title.canonical_title, authors=title.authors or ""
            )))

    all_for_dedup = prev_citations + all_citations
    clusters = deduplicate_titles(all_for_dedup)
    print(f"  Canonical titles after dedup: {len(clusters)}")

    session.query(SyllabusReference).delete()
    session.query(Title).delete()
    session.commit()

    title_map = {}
    for cluster in clusters:
        title = Title(
            canonical_title=cluster.canonical_title,
            authors=cluster.authors,
            slug=cluster.slug,
        )
        session.add(title)
        session.flush()
        title_map[cluster.slug] = title.id

    syl_id_to_citations = {}
    for syl_id, cit in all_for_dedup:
        syl_id_to_citations.setdefault(syl_id, []).append(cit)

    from pipeline.citation_extraction import normalize_title
    from rapidfuzz import fuzz, process

    cluster_slugs = [(c.slug, normalize_title(c.canonical_title)) for c in clusters]
    slug_norms = [n for _, n in cluster_slugs]

    refs_created = 0
    seen_pairs = set()
    for syl_id, cit in all_for_dedup:
        norm = normalize_title(cit.title)
        match = process.extractOne(norm, slug_norms, scorer=fuzz.token_sort_ratio, score_cutoff=80)
        if match:
            idx = slug_norms.index(match[0])
            slug = cluster_slugs[idx][0]
            title_id = title_map[slug]
            pair = (syl_id, title_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                session.add(SyllabusReference(syllabus_id=syl_id, title_id=title_id))
                refs_created += 1

    session.commit()
    print(f"  Created {refs_created} syllabus-title references")


def step_5_print_summary(session):
    """Print a summary of the database contents."""
    print(f"\n=== Database Summary ===")
    print(f"  Syllabi:    {session.query(Syllabus).count()}")
    print(f"  Topics:     {session.query(Topic).count()}")
    print(f"  Titles:     {session.query(Title).count()}")
    print(f"  References: {session.query(SyllabusReference).count()}")

    print(f"\n  Top 10 titles by reference count:")
    from sqlalchemy import func
    top_titles = (
        session.query(
            Title.canonical_title,
            Title.authors,
            func.count(SyllabusReference.syllabus_id).label("cnt"),
        )
        .join(SyllabusReference)
        .group_by(Title.id)
        .order_by(func.count(SyllabusReference.syllabus_id).desc())
        .limit(10)
        .all()
    )
    for title, authors, cnt in top_titles:
        author_str = f" — {authors}" if authors else ""
        print(f"    [{cnt:2d}] {title}{author_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", default=True,
                        help="Process sample PDFs only (default)")
    parser.add_argument("--full", action="store_true",
                        help="Process all PDFs from GCP")
    args = parser.parse_args()

    sample_only = not args.full

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = init_db()
    session = get_session()

    start = time.time()
    mode = "SAMPLE" if sample_only else "FULL"
    print(f"SyllabusLens Pipeline — {mode} mode")
    print("=" * 50)

    rows = load_csv_rows(sample_only=sample_only)
    print(f"Loaded {len(rows)} CSV rows")

    step_1_load_metadata(session, rows)
    step_2_assign_topics(session)
    step_3_extract_text(session, sample_only=sample_only)
    step_4_extract_citations(session)
    step_5_print_summary(session)

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed:.1f}s")
    session.close()


if __name__ == "__main__":
    main()
