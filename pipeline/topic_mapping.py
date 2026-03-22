"""Map subject codes from syllabi_index.csv to canonical academic topics."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CSV_PATH, DATA_DIR

# Hand-curated mapping of subject prefixes to canonical topics.
# Built by analyzing the 695 unique subject codes across OSU, UFL, UTAustin.
# Each school uses different prefix conventions; this normalizes them.
SUBJECT_TO_TOPIC = {
    # === Accounting ===
    "ACCT": "Accounting", "ACCTMIS": "Accounting", "ACG": "Accounting",
    "ACC": "Accounting", "MPA": "Accounting",

    # === Advertising / PR / Media ===
    "ADV": "Advertising", "PUR": "Public Relations",
    "MMC": "Media Communication", "JOU": "Journalism", "J": "Journalism",
    "JOUR": "Journalism", "RTV": "Radio-Television", "RTF": "Radio-Television-Film",

    # === Aerospace / Aviation ===
    "AEROENG": "Aerospace Engineering", "ASE": "Aerospace Engineering",

    # === African / African American Studies ===
    "AFAMAST": "African American Studies", "AFA": "African American Studies",
    "AAS": "African American Studies", "AFR": "African Studies",
    "AFRST": "African Studies",

    # === Agriculture ===
    "AGRCOMM": "Agriculture", "AEB": "Agriculture", "AGR": "Agriculture",
    "ANIMSCI": "Animal Science", "ANS": "Animal Science",
    "FDSCTE": "Food Science", "FOS": "Food Science", "FST": "Food Science",
    "HCS": "Horticulture", "PLNTPTH": "Plant Pathology",

    # === Anthropology ===
    "ANTHROP": "Anthropology", "ANT": "Anthropology", "ANP": "Anthropology",

    # === Architecture / Design ===
    "ARC": "Architecture", "ARCH": "Architecture",
    "ARTEDUC": "Art Education", "ARE": "Art Education",
    "ART": "Art", "AHI": "Art History", "HISTART": "Art History",
    "ARH": "Art History", "ARTSSCI": "Arts & Sciences",
    "DES": "Design", "IND": "Industrial Design", "LAR": "Landscape Architecture",
    "LARCH": "Landscape Architecture",

    # === Astronomy ===
    "ASTRON": "Astronomy", "AST": "Astronomy",

    # === Biological Sciences ===
    "BIOLOGY": "Biology", "BIO": "Biology", "BSC": "Biology", "BCH": "Biochemistry",
    "BIOCHEM": "Biochemistry", "BME": "Biomedical Engineering",
    "BIOMEDL": "Biomedical Engineering", "BMI": "Biomedical Informatics",
    "BIOETHC": "Bioethics", "BIOMSCI": "Biomedical Sciences",
    "MICROBI": "Microbiology", "MCB": "Microbiology", "M I": "Microbiology",
    "MOLGEN": "Molecular Genetics", "PCB": "Biology",
    "BOT": "Botany", "PLB": "Botany", "ECOLEV": "Ecology",
    "ZOO": "Zoology", "WIS": "Wildlife Sciences", "EEOB": "Ecology",
    "ENTOMOL": "Entomology", "ENTMLGY": "Entomology", "ENY": "Entomology",
    "ANAT": "Anatomy", "ANATOMY": "Anatomy",

    # === Business ===
    "BUSMHR": "Business", "BUSFIN": "Finance", "BUSML": "Marketing",
    "BUSOBHR": "Business", "BUSADM": "Business", "GEB": "Business",
    "MAN": "Management", "MAR": "Marketing", "FIN": "Finance",
    "FINANCE": "Finance", "RMI": "Risk Management",

    # === Chemistry ===
    "CHEM": "Chemistry", "CHM": "Chemistry", "CH": "Chemistry",

    # === Civil / Environmental Engineering ===
    "CIVILEN": "Civil Engineering", "CCE": "Civil Engineering",
    "CGN": "Civil Engineering", "C E": "Civil Engineering",
    "ENVRENG": "Environmental Engineering", "ENV": "Environmental Engineering",
    "EES": "Environmental Sciences", "EVS": "Environmental Sciences",
    "ENR": "Environment & Natural Resources", "CWR": "Water Resources",

    # === Classics / Ancient Studies ===
    "CLAS": "Classics", "CLST": "Classics", "CLA": "Classics",
    "GREEK": "Classics", "LATIN": "Classics", "GRK": "Classics", "LAT": "Classics",

    # === Communication ===
    "COMM": "Communication", "COM": "Communication", "CMS": "Communication Studies",
    "SPC": "Communication",

    # === Computer Science / IT ===
    "CSE": "Computer Science", "C S": "Computer Science", "CIS": "Computer Science",
    "COP": "Computer Science", "CDA": "Computer Science", "CNT": "Computer Science",
    "COT": "Computer Science", "CAP": "Computer Science", "CAI": "Computer Science",
    "CEN": "Computer Engineering", "ECE": "Computer Engineering",
    "COMPSTD": "Computer Science",

    # === Criminal Justice ===
    "CRIMJUS": "Criminal Justice", "CCJ": "Criminal Justice",
    "CJC": "Criminal Justice",

    # === Data Science / Statistics ===
    "STAT": "Statistics", "STA": "Statistics", "S D S": "Statistics",
    "SDS": "Statistics", "D S": "Decision Science",

    # === Dental ===
    "DENT": "Dentistry", "DEN": "Dentistry", "DNE": "Dental",

    # === Economics ===
    "ECON": "Economics", "ECO": "Economics", "ECP": "Economics", "E CO": "Economics",

    # === Education ===
    "EDUTL": "Education", "EDUCST": "Education", "EDC": "Education",
    "EDA": "Education", "EDF": "Education", "EDG": "Education",
    "EDH": "Education", "EDP": "Education", "EDS": "Education",
    "EDU": "Education", "ESE": "Education", "TSL": "Education",
    "EME": "Education Technology", "EDUPAES": "Physical Education",
    "RED": "Reading Education", "SCE": "Science Education",
    "SSE": "Social Studies Education", "SED": "Education",
    "HDFS": "Human Development", "HDF": "Human Development",

    # === Electrical / Computer Engineering ===
    "ECE": "Electrical Engineering", "E E": "Electrical Engineering",
    "EEL": "Electrical Engineering",

    # === Engineering (general) ===
    "ENG": "Engineering", "ENGI": "Engineering", "EGN": "Engineering",
    "ENGR": "Engineering", "MECHENG": "Mechanical Engineering",
    "M E": "Mechanical Engineering", "EML": "Mechanical Engineering",
    "FABENG": "Engineering", "MATSCEN": "Materials Science",
    "E M": "Engineering Mechanics", "BME": "Biomedical Engineering",
    "CBECENG": "Chemical Engineering", "CBE": "Chemical Engineering",
    "CHE": "Chemical Engineering",
    "ISE": "Industrial Engineering", "INDUSEN": "Industrial Engineering",
    "ORI": "Operations Research", "O R": "Operations Research",
    "COMPENG": "Computer Engineering",

    # === English / Writing ===
    "ENGLISH": "English", "ENC": "English", "ENG": "English",
    "ENL": "English", "AML": "English", "LIT": "English",
    "CRW": "Creative Writing", "E": "English", "RHE": "Rhetoric",
    "UWP": "Writing",

    # === Ethnic / Gender Studies ===
    "ETHNSTD": "Ethnic Studies", "WGSST": "Women's & Gender Studies",
    "WST": "Women's Studies", "WGS": "Women's & Gender Studies",
    "COMPSTD": "Comparative Studies",

    # === Geography ===
    "GEOG": "Geography", "GEO": "Geography",

    # === Geology / Earth Science ===
    "EARTHSC": "Earth Sciences", "GLY": "Geology", "GEO": "Geology",
    "ESCE": "Earth Sciences",

    # === Health / Public Health ===
    "PUBHLTH": "Public Health", "HSC": "Health Sciences",
    "PHC": "Public Health", "PUBHEPI": "Public Health",
    "HEB": "Health Education", "KIN": "Kinesiology",
    "KNSISM": "Kinesiology", "PET": "Physical Education",
    "KNSFM": "Kinesiology", "KNHES": "Kinesiology",
    "ATR": "Athletic Training", "PER": "Kinesiology",
    "PEB": "Physical Education",

    # === History ===
    "HISTORY": "History", "HIS": "History", "AMH": "History",
    "EUH": "History", "WOH": "History", "LAH": "History",
    "ASH": "History", "AFH": "History",

    # === Information Science ===
    "INFS": "Information Science", "I S": "Information Studies",
    "INF": "Information Science", "LIS": "Library Science",

    # === International Studies ===
    "INTSTDS": "International Studies", "INR": "International Relations",
    "IRG": "International Relations",

    # === Languages ===
    "FRENCH": "French", "FRE": "French", "FR": "French",
    "SPANISH": "Spanish", "SPN": "Spanish", "SP": "Spanish",
    "GERMAN": "German", "GER": "German", "GRM": "German",
    "ITALIAN": "Italian", "ITL": "Italian", "ITA": "Italian",
    "PORTGSE": "Portuguese", "POR": "Portuguese",
    "RUSSIAN": "Russian", "RUS": "Russian",
    "CHINESE": "Chinese", "CHI": "Chinese",
    "JAPANSE": "Japanese", "JPN": "Japanese",
    "KOREAN": "Korean", "KOR": "Korean",
    "ARABIC": "Arabic", "ARA": "Arabic",
    "HEBREW": "Hebrew", "HEB": "Hebrew",
    "PERSIAN": "Persian", "PRS": "Persian",
    "HINDI": "Hindi", "HIN": "Hindi",
    "URDU": "Urdu", "URD": "Urdu",
    "TURKISH": "Turkish", "TUR": "Turkish",
    "TAM": "Tamil", "TEL": "Telugu", "SAN": "Sanskrit",
    "LINGUIS": "Linguistics", "LIN": "Linguistics",
    "LANG": "Linguistics", "SLAVIC": "Slavic Studies",
    "ASL": "Sign Language",

    # === Law ===
    "LAW": "Law", "LAWFULL": "Law",

    # === Mathematics ===
    "MATH": "Mathematics", "MAC": "Mathematics", "MAD": "Mathematics",
    "MAP": "Mathematics", "MAS": "Mathematics", "MAT": "Mathematics",
    "MTH": "Mathematics", "M": "Mathematics", "MGF": "Mathematics",
    "STA": "Statistics",

    # === Medicine / Nursing ===
    "MEDCOLL": "Medicine", "MED": "Medicine", "PATHOL": "Pathology",
    "PHYSIO": "Physiology", "PHARMSCI": "Pharmacy", "PHA": "Pharmacy",
    "VETPREV": "Veterinary Medicine", "VETMED": "Veterinary Medicine",
    "VETBIOS": "Veterinary Medicine",
    "NURSING": "Nursing", "NUR": "Nursing", "NGR": "Nursing",
    "NRSADVN": "Nursing", "NRSPRCT": "Nursing",
    "NEURSGY": "Medicine", "RADONC": "Medicine",
    "OCCTHER": "Occupational Therapy", "OTH": "Occupational Therapy",
    "PHT": "Physical Therapy", "PHYSTHR": "Physical Therapy",
    "CLP": "Clinical Psychology", "MHC": "Mental Health Counseling",
    "BSN": "Nursing",

    # === Music ===
    "MUSIC": "Music", "MUS": "Music", "MUC": "Music",
    "MUE": "Music", "MUH": "Music", "MUL": "Music",
    "MUN": "Music", "MUT": "Music", "MVP": "Music",
    "MVIMG": "Music", "MVNGIMG": "Music",

    # === Philosophy ===
    "PHILOS": "Philosophy", "PHI": "Philosophy", "PHL": "Philosophy",

    # === Physics ===
    "PHYSICS": "Physics", "PHY": "Physics", "PHZ": "Physics",

    # === Political Science / Government ===
    "POLITSC": "Political Science", "POS": "Political Science",
    "GOV": "Government", "PUBAFRS": "Public Affairs",
    "PUBAFF": "Public Affairs", "PAD": "Public Administration",
    "P A": "Public Affairs",

    # === Psychology ===
    "PSYCH": "Psychology", "PSY": "Psychology", "PSB": "Psychology",
    "PCO": "Psychology", "DEP": "Psychology", "CLP": "Psychology",
    "PPE": "Psychology", "EXP": "Psychology", "SOP": "Psychology",

    # === Religion ===
    "REL": "Religion", "RELSTDS": "Religion",

    # === Social Work ===
    "SOCWORK": "Social Work", "SOW": "Social Work", "S W": "Social Work",

    # === Sociology ===
    "SOCIOL": "Sociology", "SYA": "Sociology", "SYD": "Sociology",
    "SYG": "Sociology", "SYO": "Sociology", "SYP": "Sociology",
    "SOC": "Sociology",

    # === Special Education ===
    "SPECED": "Special Education", "EEX": "Special Education",
    "SPE": "Special Education",

    # === Speech / Communication Disorders ===
    "SPHRNG": "Speech & Hearing", "SPA": "Speech Pathology",
    "SPH": "Speech & Hearing", "CSD": "Communication Disorders",
    "SPM": "Sport Management", "SSA": "Speech & Hearing",

    # === Theatre / Dance / Film / Visual Arts ===
    "THEATRE": "Theatre", "THE": "Theatre", "T D": "Theatre & Dance",
    "DAN": "Dance", "DANCE": "Dance",
    "ARTSCOL": "Art", "VAS": "Visual Art Studies",
    "VIA": "Visual Art Studies", "VIO": "Music", "VOI": "Music",

    # === Urban Planning ===
    "CRPLAN": "Urban Planning", "URP": "Urban Planning",
    "CRP": "Urban Planning",

    # === Neuroscience ===
    "NSC": "Neuroscience", "NEURO": "Neuroscience", "NEU": "Neuroscience",

    # === Nutrition ===
    "NTR": "Nutrition", "HUN": "Nutrition", "N S": "Nutritional Sciences",

    # === Marketing / Business (additional) ===
    "MKT": "Marketing", "B A": "Business", "MIS": "Management Information Systems",
    "O M": "Operations Management", "LEB": "Business Law",
    "STM": "Technology Management", "ORG": "Organizational Studies",

    # === American / Area Studies ===
    "AMS": "American Studies", "LAS": "Latin American Studies",
    "EUS": "European Studies", "MES": "Middle Eastern Studies",
    "SEA": "Southeast Asian Studies", "SAS": "South Asian Studies",
    "EAS": "East Asian Studies", "ANS": "Asian Studies",

    # === Geography (additional) ===
    "GRG": "Geography",

    # === Engineering (additional) ===
    "PGE": "Petroleum Engineering", "AET": "Applied Engineering",
    "C C": "Curriculum & Instruction", "T C": "Technical Communication",

    # === Health (additional) ===
    "HTHRHSC": "Health Sciences", "PED": "Physical Education",
    "HED": "Health Education",

    # === Textiles / Apparel ===
    "TXA": "Textiles & Apparel",

    # === Comparative Literature ===
    "C L": "Comparative Literature", "CLT": "Comparative Literature",

    # === Public Relations (additional) ===
    "P R": "Public Relations",

    # === Pharmacy (additional) ===
    "PHR": "Pharmacy",

    # === Additional Education ===
    "ALD": "Education", "CTI": "Education",

    # === Liberal Arts / Misc ===
    "L A": "Liberal Arts", "ARI": "Arts & Entertainment",
    "MNS": "Marine Science", "MARNSCI": "Marine Science",

    # === Additional unmapped high-frequency codes ===
    "P S": "Political Science", "INTR": "Interior Design",
    "BIOPHRM": "Pharmacy", "E S": "Environmental Sciences",
    "AVIATN": "Aviation", "APK": "Kinesiology",
    "PUBHHMP": "Public Health", "BUSOBA": "Business",
    "I": "Interdisciplinary Studies", "ENS": "Environmental Sciences",
    "ITD": "Interior Design", "PUBHEHS": "Public Health",
    "PBH": "Public Health", "RADSCI": "Radiology",
    "PUBHBIO": "Public Health", "SLH": "Speech & Hearing",
    "F A": "Fine Arts", "G E": "Engineering",
    "AHC": "Health Sciences", "WELDENG": "Engineering",
    "I B": "International Business", "BGS": "General Studies",
    "URB": "Urban Studies", "H S": "Health Sciences",
    "UTS": "Undergraduate Studies", "UTL": "Education",
    "M S E": "Materials Science",

    # === General / Interdisciplinary ===
    "GRADTDA": "Graduate Studies", "UGS": "Undergraduate Studies",
    "HWIH": "Honors", "IDS": "Interdisciplinary Studies",
    "CGS": "General Studies", "N": "Nursing",
    "R E": "Religious Education", "R S": "Religious Studies",
    "BDP": "Bridging Disciplines", "RIM": "Risk Management",
    "EVS": "Environmental Sciences", "J S": "Jewish Studies",
    "SYLL": None,  # not a real subject
}

# Canonical topic -> list of subtopics to merge
TOPIC_NORMALIZATION = {
    "Biology": ["Biology", "Biochemistry", "Microbiology", "Molecular Genetics",
                 "Ecology", "Botany", "Zoology", "Entomology", "Anatomy",
                 "Wildlife Sciences", "Plant Pathology"],
    "Engineering": ["Engineering", "Engineering Mechanics", "Materials Science"],
    "Music": ["Music"],
    "Medicine": ["Medicine", "Pathology", "Physiology"],
    "Health Sciences": ["Public Health", "Health Sciences", "Health Education",
                        "Kinesiology", "Athletic Training", "Physical Education",
                        "Physical Therapy", "Occupational Therapy"],
    "Education": ["Education", "Education Technology", "Physical Education",
                  "Reading Education", "Science Education",
                  "Social Studies Education", "Special Education"],
}


def build_reverse_normalization():
    """Build subtopic -> canonical_topic mapping."""
    reverse = {}
    for canonical, subs in TOPIC_NORMALIZATION.items():
        for sub in subs:
            reverse[sub] = canonical
    return reverse


def map_subject_to_topic(subject: str, department: str = "") -> str | None:
    """Map a subject code (or department) to a canonical topic name."""
    subject = subject.strip().upper()
    if not subject and department:
        subject = department.strip().upper()
    if not subject:
        return None

    topic = SUBJECT_TO_TOPIC.get(subject)
    if topic is None and subject in SUBJECT_TO_TOPIC:
        return None  # explicitly mapped to None (e.g. "SYLL")
    if topic:
        norm = build_reverse_normalization()
        return norm.get(topic, topic)
    return None


def analyze_coverage():
    """Analyze how well the mapping covers the actual CSV data."""
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    mapped = 0
    unmapped_subjects = Counter()
    topic_counts = Counter()

    for row in rows:
        subj = row["subject"].strip()
        dept = row.get("department", "").strip()
        topic = map_subject_to_topic(subj, dept)
        if topic:
            mapped += 1
            topic_counts[topic] += 1
        else:
            unmapped_subjects[subj or f"(dept={dept})"] += 1

    total = len(rows)
    print(f"Total rows: {total}")
    print(f"Mapped:     {mapped} ({100*mapped/total:.1f}%)")
    print(f"Unmapped:   {total - mapped} ({100*(total-mapped)/total:.1f}%)")

    print(f"\n=== Top 30 Topics ===")
    for topic, count in topic_counts.most_common(30):
        print(f"  {topic:35s} {count:>6,}")

    print(f"\nTotal unique topics: {len(topic_counts)}")

    print(f"\n=== Top 30 Unmapped Subjects ===")
    for subj, count in unmapped_subjects.most_common(30):
        print(f"  {subj:20s} {count:>6,}")

    return topic_counts, unmapped_subjects


def generate_topic_mapping_json():
    """Generate and save the full subject -> topic mapping."""
    norm = build_reverse_normalization()
    mapping = {}
    for subj, topic in SUBJECT_TO_TOPIC.items():
        canonical = norm.get(topic, topic)
        mapping[subj] = canonical

    out_path = DATA_DIR / "topic_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Wrote {len(mapping)} mappings to {out_path}")
    return mapping


if __name__ == "__main__":
    generate_topic_mapping_json()
    analyze_coverage()
