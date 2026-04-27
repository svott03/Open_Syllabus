"""Evaluate citation extraction precision/recall against hand-annotated ground truth.

Loads annotations from data/manually_annotations/required_readings.csv,
runs the pipeline extraction on each PDF, and compares.
"""

import csv
import re
import sys
from pathlib import Path
from dataclasses import dataclass

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from pipeline.text_extraction import extract_text
from pipeline.citation_extraction import (
    extract_citations_from_text, get_cost_summary, reset_cost_tracker,
)

ANNOTATION_DIR = DATA_DIR / "annotation_round1"
ANNOTATION_CSV = DATA_DIR / "manually_annotations" / "required_readings.csv"

CHAPTER_RE = re.compile(
    r"^(?:ch(?:apter)?\.?\s*\d|pp?\.?\s*\d|appendix|handout|lab manual|exercises? in|"
    r"text\s+\d|instructor notes$)",
    re.IGNORECASE,
)


def load_ground_truth() -> dict[str, list[str]]:
    """Load annotations CSV and return {filename: [title, ...]}."""
    gt: dict[str, list[str]] = {}
    with open(ANNOTATION_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            raw_titles = row.get("title/chapter", "").strip()
            if not raw_titles:
                continue

            titles = []
            for t in raw_titles.split(";"):
                t = t.strip().strip(",").strip()
                if not t or len(t) < 5:
                    continue
                if CHAPTER_RE.match(t):
                    continue
                titles.append(t)

            if titles:
                gt[filename] = titles
    return gt


def fuzzy_match(predicted: str, ground_truth: str, threshold: int = 60) -> bool:
    return fuzz.token_set_ratio(predicted.lower(), ground_truth.lower()) >= threshold


@dataclass
class PerFileResult:
    filename: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    unmatched_gt: list[str]
    unmatched_pred: list[str]


def evaluate_file(pdf_path: Path, gt_titles: list[str]) -> PerFileResult | None:
    """Run extraction on one PDF and compare to ground truth."""
    text, method = extract_text(str(pdf_path))
    if not text:
        print(f"  SKIP — no text extracted")
        return None

    citations = extract_citations_from_text(text)
    if citations is None:
        print(f"  SKIP — text too long ({len(text)} chars)")
        return None

    pred_titles = [c.title for c in citations]

    gt_matched = [False] * len(gt_titles)
    pred_matched = [False] * len(pred_titles)

    for i, gt in enumerate(gt_titles):
        for j, pred in enumerate(pred_titles):
            if not pred_matched[j] and fuzzy_match(pred, gt):
                gt_matched[i] = True
                pred_matched[j] = True
                break

    tp = sum(gt_matched)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return PerFileResult(
        filename=pdf_path.name, gt_count=len(gt_titles), pred_count=len(pred_titles),
        tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
        unmatched_gt=[gt_titles[i] for i, m in enumerate(gt_matched) if not m],
        unmatched_pred=[pred_titles[j] for j, m in enumerate(pred_matched) if not m],
    )


def main():
    print("=" * 70)
    print("Citation Extraction — Precision / Recall Evaluation")
    print("=" * 70)

    gt = load_ground_truth()
    print(f"Loaded {len(gt)} annotated files from {ANNOTATION_CSV.name}")

    reset_cost_tracker()
    results: list[PerFileResult] = []
    skipped_no_text = 0
    skipped_too_long = 0
    skipped_no_pdf = 0

    for filename, gt_titles in sorted(gt.items()):
        pdf_path = ANNOTATION_DIR / filename
        if not pdf_path.exists():
            skipped_no_pdf += 1
            continue

        print(f"\n--- {filename} ({len(gt_titles)} GT) ---")
        result = evaluate_file(pdf_path, gt_titles)
        if result is None:
            continue
        results.append(result)

        print(f"  Pred: {result.pred_count}  |  TP={result.tp}  FP={result.fp}  FN={result.fn}"
              f"  |  P={result.precision:.0%}  R={result.recall:.0%}")

        if result.unmatched_pred:
            for t in result.unmatched_pred[:3]:
                print(f"    FP: {t[:80]}")
            if len(result.unmatched_pred) > 3:
                print(f"    ... +{len(result.unmatched_pred) - 3} more FP")

        if result.unmatched_gt:
            for t in result.unmatched_gt[:3]:
                print(f"    FN: {t[:80]}")
            if len(result.unmatched_gt) > 3:
                print(f"    ... +{len(result.unmatched_gt) - 3} more FN")

    # Aggregate
    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)
    total_gt = sum(r.gt_count for r in results)
    total_pred = sum(r.pred_count for r in results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    macro_p = sum(r.precision for r in results) / len(results) if results else 0
    macro_r = sum(r.recall for r in results) / len(results) if results else 0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0

    cost = get_cost_summary()

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Files with annotations: {len(gt)}")
    print(f"  Files evaluated:        {len(results)}")
    if skipped_no_pdf:
        print(f"  Skipped (PDF missing):  {skipped_no_pdf}")
    print(f"  Total ground truth:     {total_gt}")
    print(f"  Total predicted:        {total_pred}")
    print(f"  TP: {total_tp}  |  FP: {total_fp}  |  FN: {total_fn}")
    print()
    print(f"  Micro  P={micro_p:.1%}  R={micro_r:.1%}  F1={micro_f1:.1%}")
    print(f"  Macro  P={macro_p:.1%}  R={macro_r:.1%}  F1={macro_f1:.1%}")
    print()
    print(f"  --- Cost ---")
    print(f"  API calls:     {cost['calls']}")
    print(f"  Input tokens:  {cost['input_tokens']:,}")
    print(f"  Output tokens: {cost['output_tokens']:,}")
    print(f"  Est. cost:     ${cost['cost_usd']:.2f}")

    print("\n--- Per-file summary ---")
    print(f"  {'File':<55} {'GT':>3} {'Pred':>4} {'P':>5} {'R':>5} {'F1':>5}")
    for r in sorted(results, key=lambda r: r.filename):
        f1 = 2 * r.precision * r.recall / (r.precision + r.recall) if (r.precision + r.recall) > 0 else 0
        print(f"  {r.filename[:53]:<55} {r.gt_count:>3} {r.pred_count:>4} {r.precision:>4.0%} {r.recall:>4.0%} {f1:>4.0%}")


if __name__ == "__main__":
    main()
