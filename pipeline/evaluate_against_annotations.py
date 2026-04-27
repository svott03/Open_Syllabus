"""Evaluate citation extraction against manually_annotations/annotations.csv.

Runs the LLM (or regex) citation extraction on each PDF in annotation_round1/,
then compares the predicted titles to the hand-annotated ground truth in
annotations.csv using fuzzy matching. Reports per-file and aggregate P/R/F1.

Results are saved to data/evaluation_results/<timestamp>_<method>/ so that
each run is preserved and never overwrites previous results.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from pipeline.text_extraction import extract_text
from pipeline.citation_extraction import (
    extract_citations_from_text, get_cost_summary, reset_cost_tracker,
    RawCitation, EXTRACTION_METHOD, MODEL_NAME,
)

ANNOTATION_DIR = DATA_DIR / "annotation_round1"
ANNOTATION_CSV = DATA_DIR / "manually_annotations" / "annotations.csv"


def load_ground_truth() -> dict[str, list[dict]]:
    """Load annotations.csv → {filename: [{title, first, last}, ...]}."""
    gt: dict[str, list[dict]] = {}
    with open(ANNOTATION_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("file") or "").strip()
            if not filename or filename == "#N/A":
                continue
            title = (row.get("title") or "").strip()
            if not title:
                continue

            gt.setdefault(filename, [])

            if title in ("None", "Error"):
                continue

            gt[filename].append({
                "title": title,
                "first": (row.get("first") or "").strip(),
                "last": (row.get("last") or "").strip(),
            })
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
    matched_pairs: list[tuple[str, str]]
    unmatched_gt: list[str]
    unmatched_pred: list[str]
    all_predictions: list[RawCitation]


def evaluate_file(pdf_path: Path, gt_entries: list[dict]) -> PerFileResult | None:
    text, method = extract_text(str(pdf_path))
    if not text:
        print(f"  SKIP — no text extracted")
        return None

    citations = extract_citations_from_text(text)
    if citations is None:
        print(f"  SKIP — text too long ({len(text)} chars)")
        return None

    pred_titles = [c.title for c in citations]
    gt_titles = [e["title"] for e in gt_entries]

    gt_matched = [False] * len(gt_titles)
    pred_matched = [False] * len(pred_titles)
    matched_pairs = []

    for i, gt in enumerate(gt_titles):
        for j, pred in enumerate(pred_titles):
            if not pred_matched[j] and fuzzy_match(pred, gt):
                gt_matched[i] = True
                pred_matched[j] = True
                matched_pairs.append((gt, pred))
                break

    tp = sum(gt_matched)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return PerFileResult(
        filename=pdf_path.name,
        gt_count=len(gt_titles),
        pred_count=len(pred_titles),
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall,
        matched_pairs=matched_pairs,
        unmatched_gt=[gt_titles[i] for i, m in enumerate(gt_matched) if not m],
        unmatched_pred=[pred_titles[j] for j, m in enumerate(pred_matched) if not m],
        all_predictions=list(citations),
    )


def save_results(
    run_dir: Path,
    method: str,
    results: list[PerFileResult],
    none_results: list[tuple[str, int, list[RawCitation]]],
    total_tp: int, total_fp: int, total_fn: int,
    total_gt: int, total_pred: int,
    micro_p: float, micro_r: float, micro_f1: float,
    macro_p: float, macro_r: float, macro_f1: float,
    none_correct: int, none_fp_total: int,
    cost: dict, elapsed: float, gt_total_files: int,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-file summary
    summary_path = run_dir / "per_file_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "gt_count", "pred_count", "tp", "fp", "fn",
                     "precision", "recall", "f1", "unmatched_gt", "unmatched_pred"])
        for r in sorted(results, key=lambda r: r.filename):
            f1 = 2 * r.precision * r.recall / (r.precision + r.recall) if (r.precision + r.recall) > 0 else 0
            w.writerow([
                r.filename, r.gt_count, r.pred_count, r.tp, r.fp, r.fn,
                round(r.precision, 4), round(r.recall, 4), round(f1, 4),
                "; ".join(r.unmatched_gt), "; ".join(r.unmatched_pred),
            ])
        for fname, pred_count, _ in sorted(none_results):
            w.writerow([fname, 0, pred_count, 0, pred_count, 0,
                         0.0 if pred_count else 1.0, 1.0, 0.0 if pred_count else 1.0,
                         "", ""])

    # 2) Detailed match log (every TP/FP/FN)
    detail_path = run_dir / "match_details.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "match_type", "gt_title", "pred_title", "pred_author"])
        for r in sorted(results, key=lambda r: r.filename):
            for gt_t, pred_t in r.matched_pairs:
                author = ""
                for c in r.all_predictions:
                    if c.title == pred_t:
                        author = c.authors or f"{c.first_name} {c.last_name}".strip()
                        break
                w.writerow([r.filename, "TP", gt_t, pred_t, author])
            for t in r.unmatched_gt:
                w.writerow([r.filename, "FN", t, "", ""])
            for t in r.unmatched_pred:
                author = ""
                for c in r.all_predictions:
                    if c.title == t:
                        author = c.authors or f"{c.first_name} {c.last_name}".strip()
                        break
                w.writerow([r.filename, "FP", "", t, author])
        for fname, pred_count, preds in sorted(none_results):
            for c in preds:
                author = c.authors or f"{c.first_name} {c.last_name}".strip()
                w.writerow([fname, "FP_NONE", "", c.title, author])

    # 3) Raw predictions (every title the pipeline found, per file)
    raw_path = run_dir / "all_predictions.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "predicted_title", "first_name", "last_name", "authors", "year"])
        for r in sorted(results, key=lambda r: r.filename):
            for c in r.all_predictions:
                w.writerow([r.filename, c.title, c.first_name, c.last_name, c.authors, c.year])
        for fname, _, preds in sorted(none_results):
            for c in preds:
                w.writerow([fname, c.title, c.first_name, c.last_name, c.authors, c.year])

    # 4) Aggregate metrics as JSON
    model = MODEL_NAME if method == "llm" else "regex"
    aggregate = {
        "run_timestamp": run_dir.name,
        "method": method,
        "model": model,
        "files_in_ground_truth": gt_total_files,
        "files_evaluated_books": len(results),
        "files_evaluated_none": len(none_results),
        "total_gt_titles": total_gt,
        "total_pred_titles": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "none_correct": none_correct,
        "none_total": len(none_results),
        "none_fp_total": none_fp_total,
        "api_calls": cost["calls"],
        "input_tokens": cost["input_tokens"],
        "output_tokens": cost["output_tokens"],
        "cost_usd": cost["cost_usd"],
        "wall_time_s": round(elapsed, 1),
    }
    aggregate_path = run_dir / "aggregate_metrics.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    return summary_path, detail_path, raw_path, aggregate_path


def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = EXTRACTION_METHOD

    print("=" * 72)
    print(f"Citation Extraction — Evaluation vs annotations.csv")
    print(f"  Method: {method}   Run: {run_ts}")
    print("=" * 72)

    gt = load_ground_truth()
    files_with_titles = {f: entries for f, entries in gt.items() if entries}
    files_none = {f for f, entries in gt.items() if not entries}

    print(f"Ground truth: {len(gt)} files total")
    print(f"  {len(files_with_titles)} files with book annotations")
    print(f"  {len(files_none)} files annotated as None/Error (no books)")

    reset_cost_tracker()
    results: list[PerFileResult] = []
    none_results: list[tuple[str, int, list[RawCitation]]] = []
    skipped_no_pdf = 0
    skipped_extraction = 0
    start = time.time()

    for filename in sorted(gt.keys()):
        pdf_path = ANNOTATION_DIR / filename
        if not pdf_path.exists():
            skipped_no_pdf += 1
            continue

        gt_entries = gt[filename]
        is_none = len(gt_entries) == 0

        if is_none:
            text, _ = extract_text(str(pdf_path))
            if not text:
                continue
            citations = extract_citations_from_text(text)
            preds = list(citations) if citations else []
            pred_count = len(preds)
            none_results.append((filename, pred_count, preds))
            label = "CORRECT" if pred_count == 0 else f"FP={pred_count}"
            print(f"  {filename[:55]:<57} None → {label}")
            continue

        print(f"\n--- {filename} ({len(gt_entries)} GT books) ---")
        result = evaluate_file(pdf_path, gt_entries)
        if result is None:
            skipped_extraction += 1
            continue
        results.append(result)

        print(f"  Pred: {result.pred_count}  |  TP={result.tp}  FP={result.fp}  FN={result.fn}"
              f"  |  P={result.precision:.0%}  R={result.recall:.0%}")

        if result.matched_pairs:
            for gt_t, pred_t in result.matched_pairs[:3]:
                if gt_t.lower()[:40] != pred_t.lower()[:40]:
                    print(f"    TP: '{gt_t[:40]}' ↔ '{pred_t[:40]}'")

        if result.unmatched_pred:
            for t in result.unmatched_pred[:5]:
                print(f"    FP: {t[:80]}")
            if len(result.unmatched_pred) > 5:
                print(f"    ... +{len(result.unmatched_pred) - 5} more FP")

        if result.unmatched_gt:
            for t in result.unmatched_gt[:5]:
                print(f"    FN: {t[:80]}")
            if len(result.unmatched_gt) > 5:
                print(f"    ... +{len(result.unmatched_gt) - 5} more FN")

    elapsed = time.time() - start

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

    none_correct = sum(1 for _, pc, _ in none_results if pc == 0)
    none_fp_total = sum(pc for _, pc, _ in none_results)

    cost = get_cost_summary()

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS (files with book annotations)")
    print("=" * 72)
    print(f"  Files in ground truth:   {len(gt)}")
    print(f"  Files evaluated (books): {len(results)}")
    print(f"  Files evaluated (none):  {len(none_results)}")
    if skipped_no_pdf:
        print(f"  Skipped (PDF missing):   {skipped_no_pdf}")
    if skipped_extraction:
        print(f"  Skipped (extraction):    {skipped_extraction}")
    print()
    print(f"  Total ground truth titles: {total_gt}")
    print(f"  Total predicted titles:    {total_pred}")
    print(f"  TP: {total_tp}  |  FP: {total_fp}  |  FN: {total_fn}")
    print()
    print(f"  Micro  P={micro_p:.1%}  R={micro_r:.1%}  F1={micro_f1:.1%}")
    print(f"  Macro  P={macro_p:.1%}  R={macro_r:.1%}  F1={macro_f1:.1%}")
    print()
    print(f"  --- 'None' files (annotated as no books) ---")
    print(f"  Correct (0 predicted):  {none_correct}/{len(none_results)}")
    print(f"  False positives total:  {none_fp_total}")
    print()
    print(f"  --- Cost ---")
    print(f"  API calls:     {cost['calls']}")
    print(f"  Input tokens:  {cost['input_tokens']:,}")
    print(f"  Output tokens: {cost['output_tokens']:,}")
    print(f"  Est. cost:     ${cost['cost_usd']:.2f}")
    print(f"  Wall time:     {elapsed:.0f}s")

    print("\n--- Per-file summary (books only) ---")
    print(f"  {'File':<57} {'GT':>3} {'Pred':>4} {'P':>5} {'R':>5} {'F1':>5}")
    for r in sorted(results, key=lambda r: r.filename):
        f1 = 2 * r.precision * r.recall / (r.precision + r.recall) if (r.precision + r.recall) > 0 else 0
        print(f"  {r.filename[:55]:<57} {r.gt_count:>3} {r.pred_count:>4} {r.precision:>4.0%} {r.recall:>4.0%} {f1:>4.0%}")

    # Save results to timestamped directory
    run_dir = DATA_DIR / "evaluation_results" / f"{run_ts}_{method}"
    paths = save_results(
        run_dir, method, results, none_results,
        total_tp, total_fp, total_fn, total_gt, total_pred,
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1,
        none_correct, none_fp_total, cost, elapsed, len(gt),
    )

    print(f"\n  Results saved to {run_dir}/")
    print(f"    per_file_summary.csv   — per-file P/R/F1")
    print(f"    match_details.csv      — every TP/FP/FN match")
    print(f"    all_predictions.csv    — all raw predictions per file")
    print(f"    aggregate_metrics.json — aggregate metrics + cost")


if __name__ == "__main__":
    main()
