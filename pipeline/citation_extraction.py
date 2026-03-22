"""Extract citations (title + author) from syllabus text.

Multi-stage pipeline:
1. Section detection: find reference/reading list sections
2. Citation parsing: regex-based structured extraction
3. Title deduplication: fuzzy matching to canonical titles
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLES_DIR
from pipeline.text_extraction import extract_text

SECTION_PATTERNS = [
    r"(?i)(?:^|\n)\s*(?:required\s+)?(?:text(?:s|book(?:s)?)?|reading(?:s)?)\s*(?:and\s+materials?)?\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*(?:course\s+)?(?:materials?|resources?)\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*(?:required|recommended|suggested)\s+(?:readings?|texts?|books?|materials?)\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*references?\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*bibliograph(?:y|ies)\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*(?:assigned\s+)?readings?\s+(?:list|schedule)\s*[:\-\n]",
    r"(?i)(?:^|\n)\s*books?\s+(?:required|for\s+(?:the\s+)?course)\s*[:\-\n]",
]

STOP_SECTIONS = [
    r"(?i)(?:^|\n)\s*(?:course\s+)?(?:schedule|calendar|outline|policies|grading|assignments?|objectives?|requirements?|expectations?|attendance|academic\s+integrity|disability|accommodations?)\s*[:\-\n]",
]


@dataclass
class RawCitation:
    title: str
    authors: str = ""
    year: str = ""
    edition: str = ""
    raw_text: str = ""


@dataclass
class CanonicalTitle:
    id: int = 0
    canonical_title: str = ""
    authors: str = ""
    slug: str = ""
    raw_citations: list[RawCitation] = field(default_factory=list)
    reference_count: int = 0


def find_reference_sections(text: str) -> list[str]:
    """Find sections of text that likely contain reading lists or references."""
    sections = []

    for pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            start = match.end()
            end = len(text)
            for stop in STOP_SECTIONS:
                stop_match = re.search(stop, text[start:])
                if stop_match:
                    end = min(end, start + stop_match.start())
            section = text[start:end].strip()
            if len(section) > 30:
                sections.append(section[:5000])

    return sections


BOOK_PATTERNS = [
    # Author (Year). Title. Publisher.
    re.compile(
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)\s*"
        r"\((?P<year>\d{4})\)\s*[.,]?\s*"
        r"(?P<title>[A-Z][^.]{10,150}?)\s*[.,]"
    ),
    # Author. Title (Year|edition). Publisher.
    re.compile(
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)\s*[.,]\s*"
        r"(?P<title>[A-Z][^.]{10,150}?)\s*"
        r"[\(,]\s*(?:(?P<year>\d{4})|(?P<edition>\d+\w*\s*ed(?:ition)?))"
    ),
    # Title by Author (possibly with ISBN or publisher after)
    re.compile(
        r"(?P<title>[A-Z][^.]{10,150}?)\s*"
        r"(?:by|,\s*by)\s+"
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)(?:\s*[.,\(]|$)"
    ),
    # "Title" - Author
    re.compile(
        r'["\u201c](?P<title>[^"\u201d]{10,150})["\u201d]\s*'
        r"[-\u2013\u2014,]\s*"
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]{3,80}?)(?:\s*[.,\(]|$)"
    ),
    # Bullet or numbered list: Title, Author (very common in syllabi)
    re.compile(
        r"(?:^|\n)\s*(?:[\u2022\u2023\u25CF\u25CB\-\*]|\d+[.\)])\s*"
        r"(?P<title>[A-Z][^,\n]{10,150}?),\s*"
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]{3,80}?)(?:\s*[.,\(]|\s*$)",
        re.MULTILINE,
    ),
]

NOISE_WORDS = {
    "the", "course", "syllabus", "university", "spring", "fall", "summer",
    "winter", "semester", "class", "section", "final", "exam", "midterm",
    "office", "hours", "grading", "policy", "attendance", "required",
    "page", "canvas", "blackboard", "students", "student", "instructor",
    "please", "assignment", "assignments", "week", "lecture", "credit",
    "disability", "services", "academic", "misconduct", "committee",
}

TITLE_BLOCKLIST_PATTERNS = [
    r"(?i)disability\s+services",
    r"(?i)academic\s+(?:misconduct|integrity|honesty)",
    r"(?i)honor\s+(?:code|pledge)",
    r"(?i)office\s+(?:hours|for\s+disability)",
    r"(?i)make\s*up\s+(?:assignment|exam|quiz)",
    r"(?i)attendance\s+polic",
    r"(?i)grading\s+(?:scale|polic|criteria)",
    r"(?i)(?:weekly|final)\s+(?:exam|update|fun-damental)",
    r"(?i)(?:download|install|access)\s+(?:r\s|python|software)",
    r"(?i)prerequisite",
    r"(?i)these\s+posts?\s+must",
    r"(?i)(?:lessons?|class(?:es)?)\s+will\s+be",
    r"(?i)(?:homework|assignment)s?\s+(?:that|require)",
    r"(?i)(?:can|will|should|must)\s+be\s+(?:submitted|approved|performed)",
]


def is_plausible_title(title: str) -> bool:
    """Filter out false-positive title extractions."""
    t = title.strip().lower()
    if len(t) < 8 or len(t) > 300:
        return False
    words = t.split()
    if len(words) < 2:
        return False
    noise_ratio = sum(1 for w in words if w in NOISE_WORDS) / len(words)
    if noise_ratio > 0.5:
        return False
    if re.match(r"^\d+$", t):
        return False
    for bp in TITLE_BLOCKLIST_PATTERNS:
        if re.search(bp, title):
            return False
    return True


def is_plausible_author(author: str) -> bool:
    """Filter out false-positive author extractions."""
    a = author.strip()
    if len(a) < 3 or len(a) > 200:
        return False
    if re.match(r"^\d+$", a):
        return False
    if a.lower() in NOISE_WORDS:
        return False
    return True


def extract_citations_from_text(text: str) -> list[RawCitation]:
    """Extract citations from syllabus text."""
    sections = find_reference_sections(text)
    if not sections:
        sections = [text[:10000]]

    citations = []
    seen_titles = set()

    for section in sections:
        for pattern in BOOK_PATTERNS:
            for match in pattern.finditer(section):
                groups = match.groupdict()
                title = (groups.get("title") or "").strip()
                authors = (groups.get("authors") or "").strip()
                year = (groups.get("year") or "").strip()
                edition = (groups.get("edition") or "").strip()

                title = re.sub(r"\s+", " ", title).strip(" .,;:")
                authors = re.sub(r"\s+", " ", authors).strip(" .,;:")

                if not is_plausible_title(title):
                    continue
                if authors and not is_plausible_author(authors):
                    authors = ""

                title_key = title.lower()[:50]
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                citations.append(RawCitation(
                    title=title, authors=authors,
                    year=year, edition=edition,
                    raw_text=match.group(0)[:200],
                ))

    return citations


def normalize_title(title: str) -> str:
    """Normalize a title for deduplication matching."""
    t = title.lower().strip()
    t = re.sub(r"\s*[:\-]\s*a\s.*$", "", t)
    t = re.sub(r"\s*\(\d+(?:st|nd|rd|th)\s+ed(?:ition)?\)\.?", "", t)
    t = re.sub(r"\s*,?\s*\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?", "", t)
    t = re.sub(r"\s*\[\s*reprint\s*\]", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def deduplicate_titles(
    all_citations: list[tuple[int, RawCitation]],
    threshold: int = 80,
) -> list[CanonicalTitle]:
    """Cluster citations into canonical titles using fuzzy matching.

    Args:
        all_citations: list of (syllabus_id, RawCitation) tuples
        threshold: rapidfuzz similarity threshold (0-100)
    """
    normalized = []
    for syl_id, cit in all_citations:
        normalized.append((normalize_title(cit.title), syl_id, cit))

    clusters: list[CanonicalTitle] = []
    cluster_keys: list[str] = []

    for norm_title, syl_id, cit in normalized:
        if cluster_keys:
            match = process.extractOne(
                norm_title, cluster_keys,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )
        else:
            match = None

        if match:
            idx = cluster_keys.index(match[0])
            clusters[idx].raw_citations.append(cit)
            clusters[idx].reference_count += 1
        else:
            cluster_keys.append(norm_title)
            clusters.append(CanonicalTitle(
                canonical_title=cit.title,
                authors=cit.authors,
                raw_citations=[cit],
                reference_count=1,
            ))

    clusters.sort(key=lambda c: c.reference_count, reverse=True)
    for i, c in enumerate(clusters):
        c.id = i + 1
        slug = re.sub(r"[^\w\s-]", "", c.canonical_title.lower())
        c.slug = re.sub(r"\s+", "-", slug)[:200]

    return clusters


def test_on_samples():
    """Run citation extraction on sample PDFs and report results."""
    all_citations = []
    syllabus_id = 0

    for school_dir in sorted(SAMPLES_DIR.iterdir()):
        if not school_dir.is_dir():
            continue
        school = school_dir.name
        pdfs = sorted(school_dir.glob("*.pdf"))
        print(f"\n--- {school.upper()} ---")
        school_citations = []

        for pdf in pdfs:
            syllabus_id += 1
            text, method = extract_text(str(pdf))
            if not text:
                print(f"  ✗ {pdf.name}: no text extracted")
                continue

            citations = extract_citations_from_text(text)
            for c in citations:
                all_citations.append((syllabus_id, c))
                school_citations.append(c)

            if citations:
                print(f"  ✓ {pdf.name}: {len(citations)} citations")
                for c in citations[:3]:
                    author_part = f" — {c.authors}" if c.authors else ""
                    print(f"      \"{c.title}\"{author_part}")
                if len(citations) > 3:
                    print(f"      ... and {len(citations) - 3} more")
            else:
                print(f"  - {pdf.name}: 0 citations found")

        print(f"  Subtotal: {len(school_citations)} citations from {len(pdfs)} PDFs")

    print(f"\n=== DEDUPLICATION ===")
    print(f"Total raw citations: {len(all_citations)}")
    clusters = deduplicate_titles(all_citations)
    print(f"Canonical titles after dedup: {len(clusters)}")
    print(f"\nTop 20 by reference count:")
    for c in clusters[:20]:
        print(f"  [{c.reference_count:2d}] {c.canonical_title}")
        if c.authors:
            print(f"       {c.authors}")

    return all_citations, clusters


if __name__ == "__main__":
    test_on_samples()
