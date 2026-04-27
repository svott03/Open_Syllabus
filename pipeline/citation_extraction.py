"""Extract citations (title + author) from syllabus text.

Supports two extraction backends:
  - "llm" (default): sends reference sections to Claude for structured extraction
  - "regex": pattern-matching fallback (48% precision, 86% recall)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLES_DIR
from pipeline.text_extraction import extract_text

EXTRACTION_METHOD = os.environ.get("CITATION_METHOD", "llm")

# ── Section detection (shared by both methods) ──────────────────────────────

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
    first_name: str = ""
    last_name: str = ""
    authors: str = ""
    year: str = ""
    edition: str = ""
    raw_text: str = ""

    def __post_init__(self):
        if not self.authors and (self.first_name or self.last_name):
            parts = [p for p in (self.first_name, self.last_name) if p]
            self.authors = " ".join(parts)


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


# ── LLM extraction (Claude via Anthropic API) ───────────────────────────────

LLM_SYSTEM_PROMPT = """\
You are a citation extraction system. You will receive the full text of a \
university course syllabus. Extract every book or textbook that is required, \
suggested, or has chapters/excerpts assigned anywhere in the syllabus \
(including the course schedule or weekly reading lists).

Scope — what to include:
- Books and textbooks listed as required readings.
- Books and textbooks listed as suggested or recommended readings.
- Books or textbooks not in the readings section but whose chapters or \
excerpts appear in the course schedule.
- If a chapter or short story is assigned as part of a larger book, list the \
larger book, not the chapter or story.

Scope — what to exclude:
- Short works such as journal articles, interviews, essays, poems, or other \
works that would conventionally be quoted (not italicized) under APA style.
- Course names, assignment descriptions, university policies, software, \
or websites.
- Instructor names (unless they are an author of an assigned book).

Title rules:
- Each book gets its own entry.
- Copy the title exactly as it appears in the syllabus, preserving the \
syllabus's own capitalization style.
- Preserve all non-English characters (umlauts, accents, macrons, etc.).
- Never include edition information in the title.
- If a title is hyphenated only because it spans a line break in the PDF, \
remove that hyphen and join the word.

Author rules:
- List only the first author of each work.
- Provide separate first_name and last_name fields.
- Mimic the name form used in the syllabus: if only a last name appears, \
leave first_name empty. Do not look up or infer names not present in the \
syllabus.
- If the author has a single name or is an institution (e.g. Aristotle, \
LUMA Institute), put it in first_name and leave last_name empty.
- Preserve non-English characters in names.

Empty syllabi:
- If no books or textbooks are found anywhere in the syllabus, return \
exactly: [{"title": "None"}]

Output format:
- Respond with ONLY a JSON array — no markdown fences, no commentary.
- Each element: {"title": "...", "first_name": "...", "last_name": "..."}
- Omit first_name or last_name when not applicable (do not use empty strings).
- If a work appears multiple times, include it only once.

Example:
[{"title": "Introduction to Algorithms", "first_name": "Thomas", \
"last_name": "Cormen"}, {"title": "Nicomachean Ethics", \
"first_name": "Aristotle"}]"""


def _get_anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it or use CITATION_METHOD=regex to fall back to regex extraction."
        )
    return anthropic.Anthropic(api_key=api_key)


MAX_TEXT_CHARS = 100_000

COST_PER_1M_INPUT = {
    "claude-opus-4-6": 15.00,
    "claude-sonnet-4-20250514": 3.00,
    "claude-haiku-3-5-20241022": 0.80,
}
COST_PER_1M_OUTPUT = {
    "claude-opus-4-6": 75.00,
    "claude-sonnet-4-20250514": 15.00,
    "claude-haiku-3-5-20241022": 4.00,
}
MODEL_NAME = "claude-sonnet-4-20250514"

_cost_tracker = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_cost_summary() -> dict:
    """Return cumulative token usage and estimated cost."""
    inp = _cost_tracker["input_tokens"]
    out = _cost_tracker["output_tokens"]
    cost_in = inp / 1_000_000 * COST_PER_1M_INPUT.get(MODEL_NAME, 0)
    cost_out = out / 1_000_000 * COST_PER_1M_OUTPUT.get(MODEL_NAME, 0)
    return {
        "calls": _cost_tracker["calls"],
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": round(cost_in + cost_out, 4),
    }


def reset_cost_tracker():
    _cost_tracker["input_tokens"] = 0
    _cost_tracker["output_tokens"] = 0
    _cost_tracker["calls"] = 0


def extract_citations_llm(text: str, max_retries: int = 2) -> list[RawCitation] | None:
    """Extract citations using Claude.

    Sends the full syllabus text so that citations embedded in course
    schedules or weekly reading lists are not missed.

    Returns None if the text exceeds MAX_TEXT_CHARS (caller should skip).
    """
    if len(text) > MAX_TEXT_CHARS:
        return None

    content = text

    client = _get_anthropic_client()

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2048,
                system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            _cost_tracker["input_tokens"] += response.usage.input_tokens
            _cost_tracker["output_tokens"] += response.usage.output_tokens
            _cost_tracker["calls"] += 1
            raw = response.content[0].text.strip()

            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            citations_data = json.loads(raw)
            if not isinstance(citations_data, list):
                return []

            citations = []
            seen = set()
            for item in citations_data:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                if title == "None":
                    return []
                if len(title) < 5:
                    continue
                key = title.lower()[:60]
                if key in seen:
                    continue
                seen.add(key)

                first = (item.get("first_name") or "").strip()
                last = (item.get("last_name") or "").strip()
                authors = (item.get("authors") or "").strip()

                citations.append(RawCitation(
                    title=title,
                    first_name=first,
                    last_name=last,
                    authors=authors,
                    year=(item.get("year") or "").strip(),
                ))
            return citations

        except json.JSONDecodeError:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            if "rate_limit" in str(e).lower() or "overloaded" in str(e).lower():
                time.sleep(5 * (attempt + 1))
                if attempt < max_retries:
                    continue
            raise


# ── Regex extraction (fallback) ─────────────────────────────────────────────

BOOK_PATTERNS = [
    re.compile(
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)\s*"
        r"\((?P<year>\d{4})\)\s*[.,]?\s*"
        r"(?P<title>[A-Z][^.]{10,150}?)\s*[.,]"
    ),
    re.compile(
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)\s*[.,]\s*"
        r"(?P<title>[A-Z][^.]{10,150}?)\s*"
        r"[\(,]\s*(?:(?P<year>\d{4})|(?P<edition>\d+\w*\s*ed(?:ition)?))"
    ),
    re.compile(
        r"(?P<title>[A-Z][^.]{10,150}?)\s*"
        r"(?:by|,\s*by)\s+"
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]+?)(?:\s*[.,\(]|$)"
    ),
    re.compile(
        r'["\u201c](?P<title>[^"\u201d]{10,150})["\u201d]\s*'
        r"[-\u2013\u2014,]\s*"
        r"(?P<authors>[A-Z][a-zA-Z\.\-\s,&]{3,80}?)(?:\s*[.,\(]|$)"
    ),
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
    a = author.strip()
    if len(a) < 3 or len(a) > 200:
        return False
    if re.match(r"^\d+$", a):
        return False
    if a.lower() in NOISE_WORDS:
        return False
    return True


def extract_citations_regex(text: str) -> list[RawCitation]:
    """Extract citations using regex patterns (fallback method)."""
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


# ── Public API ──────────────────────────────────────────────────────────────

def extract_citations_from_text(text: str) -> list[RawCitation] | None:
    """Extract citations using the configured method (LLM or regex).

    Returns None if the text is too long and should be skipped.
    """
    method = EXTRACTION_METHOD
    if method == "llm":
        try:
            return extract_citations_llm(text)
        except RuntimeError:
            print("  [warn] LLM unavailable, falling back to regex")
            return extract_citations_regex(text)
    return extract_citations_regex(text)


# ── Deduplication ───────────────────────────────────────────────────────────

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
    """Cluster citations into canonical titles using fuzzy matching."""
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


# ── CLI test ────────────────────────────────────────────────────────────────

def test_on_samples():
    """Run citation extraction on sample PDFs and report results."""
    method = EXTRACTION_METHOD
    print(f"Extraction method: {method}")
    if method == "llm":
        try:
            _get_anthropic_client()
            print("Anthropic API key: OK")
        except RuntimeError as e:
            print(f"Anthropic API key: MISSING — {e}")
            return

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
            text, method_used = extract_text(str(pdf))
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
