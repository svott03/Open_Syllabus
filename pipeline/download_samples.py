"""Download a stratified sample of syllabus PDFs from GCP for testing."""

import csv
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CSV_PATH, SAMPLES_DIR, GCP_BUCKET, SCHOOLS, SAMPLES_PER_SCHOOL


def load_csv_by_school():
    """Load CSV and group rows by university, filtering to rows with local_path."""
    by_school = {s: [] for s in SCHOOLS}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            uni = row["university"]
            if uni in by_school and row["local_path"].strip():
                by_school[uni].append(row)
    return by_school


def pick_diverse_sample(rows, n, school):
    """Pick n rows with diverse subjects. Includes edge cases for UFL."""
    subjects = {}
    for r in rows:
        subj = r["subject"].strip() or "__EMPTY__"
        subjects.setdefault(subj, []).append(r)

    picked = []
    subject_list = list(subjects.keys())
    random.seed(42)
    random.shuffle(subject_list)

    for subj in subject_list:
        if len(picked) >= n:
            break
        pool = subjects[subj]
        picked.append(random.choice(pool))

    if len(picked) < n:
        remaining = [r for r in rows if r not in picked]
        random.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])

    return picked[:n]


def download_pdf(local_path, dest_dir):
    """Download a single PDF from the GCP bucket."""
    gs_path = f"gs://{GCP_BUCKET}/{local_path}"
    dest = dest_dir / Path(local_path).name
    if dest.exists():
        return dest, True
    result = subprocess.run(
        ["gsutil", "-q", "cp", gs_path, str(dest)],
        capture_output=True, text=True,
    )
    return dest, result.returncode == 0


def main():
    by_school = load_csv_by_school()
    total_downloaded = 0
    total_failed = 0

    for school in SCHOOLS:
        rows = by_school[school]
        sample = pick_diverse_sample(rows, SAMPLES_PER_SCHOOL, school)
        dest_dir = SAMPLES_DIR / school
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {school.upper()} ({len(sample)} samples from {len(rows)} total) ---")

        for i, row in enumerate(sample):
            dest, ok = download_pdf(row["local_path"], dest_dir)
            status = "OK" if ok else "FAIL"
            if ok:
                total_downloaded += 1
            else:
                total_failed += 1
            subj = row["subject"] or "(no subject)"
            print(f"  [{i+1}/{len(sample)}] {status} {subj:12s} {row['local_path']}")

        manifest = dest_dir / "sample_manifest.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sample[0].keys())
            writer.writeheader()
            writer.writerows(sample)
        print(f"  Manifest written to {manifest}")

    print(f"\nDone: {total_downloaded} downloaded, {total_failed} failed")


if __name__ == "__main__":
    main()
