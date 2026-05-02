"""Seed the database with annotation-round1 LLM-extracted citations.

Reads annotation_round1_files.csv for syllabus metadata, looks up full
metadata from syllabi_index.csv, and loads LLM predictions from
evaluation_results into the Title + SyllabusReference tables.
"""

import csv
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, CSV_PATH
from db.models import init_db, get_session, Syllabus, Topic, Title, SyllabusReference
from pipeline.topic_mapping import map_subject_to_topic
from pipeline.citation_extraction import (
    RawCitation, normalize_title, deduplicate_titles,
)

ANNOTATION_FILES_CSV = DATA_DIR / "annotation_round1_files.csv"
PREDICTIONS_CSV = DATA_DIR / "evaluation_results" / "20260426_204820_llm" / "all_predictions.csv"


def load_annotation_metadata() -> list[dict]:
    """Load annotation_round1_files.csv and enrich with syllabi_index.csv."""
    ann_rows = []
    with open(ANNOTATION_FILES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ann_rows.append(row)

    index_lookup = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lp = row.get("local_path", "").strip()
                if lp:
                    index_lookup[lp] = row

    enriched = []
    for row in ann_rows:
        local_path = row["local_path"].strip()
        full = index_lookup.get(local_path, {})
        enriched.append({
            "university": row["university"].strip(),
            "subject": row["subject"].strip(),
            "local_path": local_path,
            "college": full.get("college", ""),
            "department": full.get("department", ""),
            "course_number": full.get("course_number", ""),
            "course_title": full.get("course_title", ""),
            "term": full.get("term", ""),
            "term_label": full.get("term_label", ""),
            "section": full.get("section", ""),
            "instructor": full.get("instructor", ""),
            "url": full.get("url", ""),
        })
    return enriched


def load_predictions() -> dict[str, list[RawCitation]]:
    """Load all_predictions.csv → {filename: [RawCitation, ...]}."""
    preds: dict[str, list[RawCitation]] = {}
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row["file"].strip()
            title = row["predicted_title"].strip()
            if not title:
                continue
            preds.setdefault(filename, []).append(RawCitation(
                title=title,
                first_name=row.get("first_name", "").strip(),
                last_name=row.get("last_name", "").strip(),
                authors=row.get("authors", "").strip(),
                year=row.get("year", "").strip(),
            ))
    return preds


def filename_from_local_path(local_path: str) -> str:
    """Convert 'osu/View_Syllabi_xxx.pdf' → 'osu_View_Syllabi_xxx.pdf'."""
    return local_path.replace("/", "_")


def main():
    print("=" * 60)
    print("Seed DB with annotation-round citations")
    print("=" * 60)

    init_db()
    session = get_session()

    # Step 1: Insert annotation syllabus rows
    metadata = load_annotation_metadata()
    existing_paths = {r[0] for r in session.query(Syllabus.local_path).all()}

    topic_cache = {}
    for t in session.query(Topic).all():
        topic_cache[t.name] = t

    inserted = 0
    for row in metadata:
        if row["local_path"] in existing_paths:
            continue

        topic_name = map_subject_to_topic(row["subject"], row.get("department", ""))
        topic = None
        if topic_name:
            if topic_name not in topic_cache:
                topic = Topic(name=topic_name)
                session.add(topic)
                session.flush()
                topic_cache[topic_name] = topic
            topic = topic_cache[topic_name]

        syl = Syllabus(
            university=row["university"],
            college=row["college"],
            department=row["department"],
            course_number=row["course_number"],
            course_title=row["course_title"],
            subject=row["subject"],
            term=row["term"],
            term_label=row["term_label"],
            section=row["section"],
            instructor=row["instructor"],
            url=row["url"],
            local_path=row["local_path"],
            topic_id=topic.id if topic else None,
        )
        session.add(syl)
        inserted += 1

    session.commit()
    print(f"  Inserted {inserted} new syllabi ({len(metadata) - inserted} already existed)")

    # Step 2: Build local_path → syllabus_id mapping
    path_to_id = {}
    for syl_id, lp in session.query(Syllabus.id, Syllabus.local_path).all():
        path_to_id[lp] = syl_id

    # Step 3: Load predictions and build citation list
    predictions = load_predictions()
    print(f"  Loaded predictions for {len(predictions)} files")

    # Step 4: Gather ALL citations (existing DB + new predictions)
    all_citations = []

    for ref in session.query(SyllabusReference).all():
        title = session.get(Title, ref.title_id)
        if title:
            all_citations.append((ref.syllabus_id, RawCitation(
                title=title.canonical_title, authors=title.authors or ""
            )))

    new_count = 0
    for filename, cits in predictions.items():
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            school = parts[0].split("_")[0]
            rest = filename[len(school) + 1:]
            local_path = f"{school}/{rest}"
        else:
            local_path = filename

        syl_id = path_to_id.get(local_path)
        if not syl_id:
            continue

        for cit in cits:
            all_citations.append((syl_id, cit))
            new_count += 1

    print(f"  Total citations for dedup: {len(all_citations)} ({new_count} new)")

    # Step 5: Deduplicate and rebuild Title + SyllabusReference tables
    clusters = deduplicate_titles(all_citations)
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

    slug_norms = [(c.slug, normalize_title(c.canonical_title)) for c in clusters]
    norm_list = [n for _, n in slug_norms]

    refs_created = 0
    seen_pairs = set()
    for syl_id, cit in all_citations:
        norm = normalize_title(cit.title)
        match = process.extractOne(
            norm, norm_list, scorer=fuzz.token_sort_ratio, score_cutoff=80
        )
        if match:
            idx = norm_list.index(match[0])
            slug = slug_norms[idx][0]
            tid = title_map[slug]
            pair = (syl_id, tid)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                session.add(SyllabusReference(syllabus_id=syl_id, title_id=tid))
                refs_created += 1

    session.commit()
    print(f"  Created {refs_created} syllabus-title references")

    # Summary
    print(f"\n  DB now contains:")
    print(f"    Syllabi:    {session.query(Syllabus).count()}")
    print(f"    Topics:     {session.query(Topic).count()}")
    print(f"    Titles:     {session.query(Title).count()}")
    print(f"    References: {session.query(SyllabusReference).count()}")

    from sqlalchemy import func
    print(f"\n  Top 10 titles by reference count:")
    top = (
        session.query(Title.canonical_title, Title.authors,
                      func.count(SyllabusReference.syllabus_id).label("cnt"))
        .join(SyllabusReference)
        .group_by(Title.id)
        .order_by(func.count(SyllabusReference.syllabus_id).desc())
        .limit(10)
        .all()
    )
    for t, a, c in top:
        author = f" — {a}" if a else ""
        print(f"    [{c:2d}] {t}{author}")

    session.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
