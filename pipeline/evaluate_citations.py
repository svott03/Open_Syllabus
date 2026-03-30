"""Evaluate citation extraction precision/recall against hand-annotated ground truth.

Reads 10 annotation PDFs, runs the pipeline extraction, and compares
against manually identified citations from each syllabus.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from pipeline.text_extraction import extract_text
from pipeline.citation_extraction import extract_citations_from_text

ANNOTATION_DIR = DATA_DIR / "annotation_round1"

GROUND_TRUTH: dict[str, list[str]] = {
    "osu_View_Syllabi_5280bb2a.pdf": [
        # MOLGEN 4500 — Molecular Genetics
        "Concepts of Genetics",
    ],
    "osu_View_Syllabi_60b45aae.pdf": [
        # MEDDIET 5189/5289/5389 — Medical Dietetics
        "Nutrition Therapy and Pathophysiology",
        "Academy of Nutrition and Dietetics Pocket Guide to Nutrition Assessment",
    ],
    "osu_View_Syllabi_05bb2ad4.pdf": [
        # SOCWORK 4503 — Social Work
        "Promoting Community Change: Making It Happen in the Real World",
    ],
    "osu_View_Syllabi_dafee0ae.pdf": [
        # FMST 4895 — Film Studies, Digital Cinema
        "The Virtual Life of Film",
        "The Ontology of the Photographic Image",
        "An Aesthetic of Reality",
        "Bicycle Thief",
        "Logic as Semiotic",
        "The Tactile Eye",
        "What is Digital Cinema?",
        "What My Fingers Knew",
        "The Devil Finds Work",
        "The Bastard Spawn",
        "The Skin and the Screen",
        "Realist Film Theory",
        "A Critical Theory of Binge-Watching",
        "Television Studies Goes Digital",
        "Toward a Phenomenology of Nonfictional Film Experience",
        "Neo-Neo Realism",
        "The Waning of Narrative",
        "The End of Cinema",
    ],
    "osu_View_Syllabi_9b5f61ea.pdf": [
        # BIOCHEM 5621 — Biochemistry Lab
        "Biochemistry Laboratory, Modern Theory and Techniques",
    ],
    "osu_View_Syllabi_665f29c5.pdf": [
        # ARCH 5590 — Revit
        "Autodesk Revit 2022: Fundamentals for Architecture",
        "Mastering Autodesk Revit 2020",
    ],
    "osu_View_Syllabi_e85822f2.pdf": [
        # PUBHEPI 6440 — Reproductive/Perinatal Epidemiology
        "Fertility and Pregnancy: An Epidemiologic Perspective",
        "Terms in reproductive and perinatal epidemiology: I. Reproductive terms",
        "Terms in reproductive and perinatal epidemiology: I. Perinatal terms",
        "The use of United States vital statistics in perinatal and obstetrics research",
        "Caffeine and miscarriage risk",
        "Maternal caffeine consumption during pregnancy and the risk of miscarriage",
        "Obstetric and perinatal outcomes in singleton pregnancies resulting from the transfer of frozen thawed versus fresh embryos",
        "Examining Infertility Treatment and Early Childhood Development in the Upstate KIDS Study",
        "Causal Diagrams for Epidemiologic Research",
        "The Birth Weight Paradox Uncovered",
        "Diagnosis and Management of Endometriosis",
        "Uterine Fibroid Tumors: Diagnosis and Treatment",
        "Evaluation and Management of Abnormal Uterine Bleeding in Premenopausal Women",
        "Weight gain during pregnancy: reexamining the guidelines",
        "The bias in current measures of gestational weight gain",
        "Routine weighing of women during pregnancy is of limited value and should be abandoned",
        "Case-crossover and case-time-control designs in birth defects epidemiology",
        "Triggers of spontaneous preterm delivery--why today?",
        "Prepregnancy body mass index and gestational weight gain in relation to child body mass index among siblings",
        "Normal labor and delivery",
        "Multiple gestations",
    ],
    "osu_View_Syllabi_fe4f9907.pdf": [
        # AMIS 3200 — Intermediate Accounting
        "Intermediate Accounting",
    ],
    "osu_View_Syllabi_a2c76991.pdf": [
        # MDRNGRK 2000 — Athens: The Modern City
        "Athens",  # Llewellyn Smith book on Athens (exact title unclear, referenced as "Llewellyn Smith")
        "Heirs of the Greek Catastrophe",
        "Eurydice Street: A Place in Athens",
        "Inside Hitler's Greece",
        "This Child Died Tomorrow",
        "A Short Border Handbook",
        "The Positive Aspects of Greek Urbanization",
        "The Aesthetic Impact of Graffiti",
        "Camaraderie in the Face of Greek Austerity",
        "Selling Greece",
        "The Rise of Golden Dawn",
    ],
    "osu_View_Syllabi_63b6d8cc.pdf": [
        # GEOSCIM 7875 — Spectral Methods in Geodesy
        "Spectral Methods: Fundamentals in Single Domains",
        "Weighting algorithms to stack superconducting gravimeter data",
        "A search for the Slichter modes in superconducting gravimeter records",
        "First observation of 2S1 and study of the splitting of the football mode",
        "Multiple-taper spectral analysis: A stand-alone C-subroutine",
        "Models of the Lateral Heterogeneity of the Earth Consistent with Eigenfrequency Splitting Data",
        "A spectral search for the inner core wobble in Earth's polar motion",
        "Application of the cos-Fourier expansion to data transformation",
        "On the Inductive Proof of Legendre Addition Theorem",
        "Computing Fourier transforms and convolutions on the 2-sphere",
        "Computational Harmonic Analysis for Tensor Fields on the Two-Sphere",
        "Green's Function of Earth's Deformation as a Result of Atmospheric Loading",
        "Spectral analysis using orthonormal functions",
        "Improved regional gravity fields on the Moon from Lunar Prospector tracking data",
    ],
}


def fuzzy_match(predicted: str, ground_truth: str, threshold: int = 60) -> bool:
    """Check if a predicted title fuzzy-matches a ground truth title."""
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


@dataclass
class PerFileResult:
    filename: str
    subject: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    matched_gt: list[str]
    unmatched_gt: list[str]
    unmatched_pred: list[str]


def evaluate_file(pdf_path: Path, gt_titles: list[str]) -> PerFileResult:
    """Run extraction on one PDF and compare to ground truth."""
    text, method = extract_text(str(pdf_path))
    if not text:
        return PerFileResult(
            filename=pdf_path.name, subject="", gt_count=len(gt_titles),
            pred_count=0, tp=0, fp=0, fn=len(gt_titles),
            precision=0.0, recall=0.0,
            matched_gt=[], unmatched_gt=gt_titles, unmatched_pred=[],
        )

    citations = extract_citations_from_text(text)
    if citations is None:
        print(f"  SKIPPED — text too long ({len(text)} chars)")
        return PerFileResult(
            filename=pdf_path.name, subject="", gt_count=len(gt_titles),
            pred_count=0, tp=0, fp=0, fn=len(gt_titles),
            precision=0.0, recall=0.0,
            matched_gt=[], unmatched_gt=gt_titles, unmatched_pred=[],
        )
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

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    matched = [gt_titles[i] for i, m in enumerate(gt_matched) if m]
    unmatched_gt_list = [gt_titles[i] for i, m in enumerate(gt_matched) if not m]
    unmatched_pred_list = [pred_titles[j] for j, m in enumerate(pred_matched) if not m]

    return PerFileResult(
        filename=pdf_path.name, subject=pdf_path.stem,
        gt_count=len(gt_titles), pred_count=len(pred_titles),
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall,
        matched_gt=matched, unmatched_gt=unmatched_gt_list,
        unmatched_pred=unmatched_pred_list,
    )


def main():
    print("=" * 70)
    print("Citation Extraction — Precision / Recall Evaluation")
    print("=" * 70)

    results: list[PerFileResult] = []
    total_tp = total_fp = total_fn = 0

    for filename, gt_titles in GROUND_TRUTH.items():
        pdf_path = ANNOTATION_DIR / filename
        if not pdf_path.exists():
            print(f"\n  SKIP {filename}: file not found")
            continue

        print(f"\n--- {filename} ({len(gt_titles)} ground truth) ---")
        result = evaluate_file(pdf_path, gt_titles)
        results.append(result)
        total_tp += result.tp
        total_fp += result.fp
        total_fn += result.fn

        print(f"  Extracted: {result.pred_count}  |  TP={result.tp}  FP={result.fp}  FN={result.fn}")
        print(f"  Precision: {result.precision:.0%}  |  Recall: {result.recall:.0%}")

        if result.unmatched_pred:
            print(f"  False positives (predicted but not in GT):")
            for t in result.unmatched_pred[:5]:
                print(f"    - {t[:80]}")
            if len(result.unmatched_pred) > 5:
                print(f"    ... and {len(result.unmatched_pred) - 5} more")

        if result.unmatched_gt:
            print(f"  Missed (in GT but not extracted):")
            for t in result.unmatched_gt[:5]:
                print(f"    - {t[:80]}")
            if len(result.unmatched_gt) > 5:
                print(f"    ... and {len(result.unmatched_gt) - 5} more")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    total_gt = sum(r.gt_count for r in results)
    total_pred = sum(r.pred_count for r in results)
    macro_precision = sum(r.precision for r in results) / len(results) if results else 0
    macro_recall = sum(r.recall for r in results) / len(results) if results else 0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0)
    macro_f1 = (2 * macro_precision * macro_recall / (macro_precision + macro_recall)
                if (macro_precision + macro_recall) > 0 else 0)

    print(f"  Files evaluated:       {len(results)}")
    print(f"  Total ground truth:    {total_gt}")
    print(f"  Total predicted:       {total_pred}")
    print(f"  True positives:        {total_tp}")
    print(f"  False positives:       {total_fp}")
    print(f"  False negatives:       {total_fn}")
    print()
    print(f"  Micro precision:       {micro_precision:.1%}")
    print(f"  Micro recall:          {micro_recall:.1%}")
    print(f"  Micro F1:              {micro_f1:.1%}")
    print()
    print(f"  Macro precision:       {macro_precision:.1%}")
    print(f"  Macro recall:          {macro_recall:.1%}")
    print(f"  Macro F1:              {macro_f1:.1%}")

    print("\n--- Per-file summary ---")
    print(f"  {'File':<45} {'GT':>3} {'Pred':>4} {'P':>6} {'R':>6} {'F1':>6}")
    for r in results:
        f1 = (2 * r.precision * r.recall / (r.precision + r.recall)
              if (r.precision + r.recall) > 0 else 0)
        short = r.filename[:43]
        print(f"  {short:<45} {r.gt_count:>3} {r.pred_count:>4} {r.precision:>5.0%} {r.recall:>5.0%} {f1:>5.0%}")


if __name__ == "__main__":
    main()
