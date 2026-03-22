import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "syllabus.db"
CSV_PATH = BASE_DIR / "syllabi_index.csv"

GCP_BUCKET = "syllabus-480300-syllabi"
GCP_PROJECT = "syllabus-480300"

SCHOOLS = ["osu", "ufl", "utaustin"]
SCHOOL_LABELS = {"osu": "Ohio State University", "ufl": "University of Florida", "utaustin": "UT Austin"}

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
SYNC_DATABASE_URL = f"sqlite:///{DB_PATH}"

SAMPLES_PER_SCHOOL = 35
