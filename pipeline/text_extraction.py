"""Extract text from syllabus PDFs using PyMuPDF with pdfplumber fallback."""

import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLES_DIR


def extract_with_pymupdf(pdf_path: str) -> str | None:
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages).strip()
        return text if len(text) > 50 else None
    except Exception:
        return None


def extract_with_pdfplumber(pdf_path: str) -> str | None:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages).strip()
        return text if len(text) > 50 else None
    except Exception:
        return None


def extract_text(pdf_path: str) -> tuple[str | None, str]:
    """Extract text from a PDF. Returns (text, method_used)."""
    text = extract_with_pymupdf(pdf_path)
    if text:
        return text, "pymupdf"
    text = extract_with_pdfplumber(pdf_path)
    if text:
        return text, "pdfplumber"
    return None, "failed"


def test_on_samples():
    """Run extraction on all downloaded samples and report results."""
    results = {"pymupdf": 0, "pdfplumber": 0, "failed": 0}
    details = []

    for school_dir in sorted(SAMPLES_DIR.iterdir()):
        if not school_dir.is_dir():
            continue
        school = school_dir.name
        pdfs = sorted(school_dir.glob("*.pdf"))
        print(f"\n--- {school.upper()} ({len(pdfs)} files) ---")

        for pdf in pdfs:
            text, method = extract_text(str(pdf))
            results[method] += 1
            char_count = len(text) if text else 0
            details.append({
                "school": school, "file": pdf.name,
                "method": method, "chars": char_count,
            })
            indicator = "✓" if method != "failed" else "✗"
            print(f"  {indicator} {pdf.name:45s} {method:12s} {char_count:>7,} chars")

    print(f"\n=== SUMMARY ===")
    total = sum(results.values())
    for method, count in results.items():
        pct = 100 * count / total if total else 0
        print(f"  {method:12s}: {count:3d} ({pct:.1f}%)")

    failed_files = [d for d in details if d["method"] == "failed"]
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for d in failed_files:
            print(f"  {d['school']}/{d['file']}")

    return details


if __name__ == "__main__":
    test_on_samples()
