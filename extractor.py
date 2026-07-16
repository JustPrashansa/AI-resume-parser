import json
import re
import time
import random
import pdfplumber
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
OPENAI_MODEL = "gpt-4.1-mini"

def sanitize_age(value):
    """
    The LLM sometimes returns a birthdate, a range, or other garbage in
    the 'age' field instead of a plain integer. Only accept something
    that's actually a plausible human age; everything else becomes None
    rather than polluting the data with junk like "January 02, 1987".
    Always returns a real Python int (never a float/str), or None.
    """

    if value is None:
        return None

    try:
        age_int = int(float(str(value).strip()))

        if 15 <= age_int <= 80:
            return age_int

    except (ValueError, TypeError):
        pass

    return None


def sanitize_experience(value):
    """Coerce experience_years to a real int (not float/str), or None."""

    if value is None:
        return None

    try:
        return int(float(str(value).strip()))

    except (ValueError, TypeError):
        return None


def sanitize_college_type(value):
    """Only 'IIT', 'NIT', or None are valid — guard against model drift."""

    if not value:
        return None

    value_upper = str(value).strip().upper()

    if "IIT" in value_upper:
        return "IIT"

    if "NIT" in value_upper:
        return "NIT"

    return None


def sanitize_phone(value):
    """
    Normalize every phone number to the SAME format: +91XXXXXXXXXX.
    Handles numbers that already have +91, 91, spaces, dashes, or nothing.
    Returns None if it doesn't look like a valid 10-digit Indian mobile
    number after cleanup, rather than keeping garbage.
    """

    if not value:
        return None

    digits = re.sub(r"\D", "", str(value))

    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]

    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    if len(digits) == 10 and digits[0] in "6789":
        return f"+91{digits}"

    return None


def flatten_value(item):
    """
    The LLM is inconsistent about whether education_history/college/
    previous_institutions entries are plain strings or nested objects
    with arbitrary keys. This forces everything into a single readable
    string so every record in the cache/Excel has the same shape.
    """

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        parts = [
            f"{k}: {v}"
            for k, v in item.items()
            if v not in (None, "", [])
        ]
        return "; ".join(parts) if parts else None

    if isinstance(item, list):
        flattened = [flatten_value(x) for x in item]
        flattened = [x for x in flattened if x]
        return " / ".join(flattened) if flattened else None

    return str(item)


def flatten_list_field(value):
    """Apply flatten_value to every item in a list field, drop empties."""

    if not isinstance(value, list):
        return []

    result = [flatten_value(item) for item in value]
    return [item for item in result if item]


def read_pdf(file, char_limit=8000):
    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

            if len(text) >= char_limit:
                break

    return text


def read_docx(file):

    doc = Document(file)

    return "\n".join(
        para.text
        for para in doc.paragraphs
    )


def extract_email(text):

    match = re.search(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        text
    )

    return match.group(0) if match else None


def extract_phone(text):

    match = re.search(
        r'(\+91[- ]?)?[6-9]\d{9}',
        text
    )

    return match.group(0) if match else None


def extract_experience(text):

    matches = []

    matches.extend(
        re.findall(
            r'(\d+)\+?\s*years',
            text,
            re.IGNORECASE
        )
    )

    matches.extend(
        re.findall(
            r'(\d+)\+?\s*yrs',
            text,
            re.IGNORECASE
        )
    )

    matches.extend(
        re.findall(
            r'total\s*yrs?.*?(\d+)',
            text,
            re.IGNORECASE
        )
    )

    if matches:

        try:
            return max(
                int(x)
                for x in matches
            )

        except Exception:
            pass

    return None


def extract_qualifications(text):

    qualifications = []

    keywords = [
        "PhD",
        "Ph.D",
        "Ph. D",
        "Doctor of Philosophy",
        "UGC NET",
        "NET",
        "CSIR-NET",
        "GATE",
        "JRF",

        "MBA",
        "PGDM",
        "M.Tech",
        "M.Ed",
        "M.Sc",
        "M.A",
        "M.Com",
        "MCA",
        "MBM",

        "B.Tech",
        "B.Ed",
        "B.Sc",
        "B.A",
        "B.Com",
        "BCA",
        "BBA",

        "D.El.Ed",
        "D.T.Ed",
        "DSM",

        "CTET",
        "TET"
    ]

    lower_text = text.lower()

    for keyword in keywords:

        if keyword.lower() in lower_text:
            qualifications.append(keyword)

    return qualifications

_QUALIFICATION_RANK = {
    "PhD": 100,
    "Ph.D": 100,
    "Ph. D": 100,
    "Doctor of Philosophy": 100,
    "UGC NET": 95,
    "CSIR-NET": 95,
    "NET": 95,
    "JRF": 95,

    "GATE": 90,

    "MBA": 85,
    "PGDM": 85,
    "M.Tech": 85,
    "MCA": 85,

    "MBM": 80,
    "M.Ed": 80,
    "M.Sc": 80,
    "M.A": 80,
    "M.Com": 80,

    "B.Tech": 70,
    "B.Ed": 70,
    "B.Sc": 70,
    "B.A": 70,
    "B.Com": 70,
    "BCA": 70,
    "BBA": 70,

    "D.El.Ed": 60,
    "D.T.Ed": 60,
    "DSM": 50,

    "CTET": 40,
    "TET": 40
}


def split_qualifications(qualification_list):

    if not qualification_list:
        return None, []

    ranked = sorted(
        qualification_list,
        key=lambda q: _QUALIFICATION_RANK.get(q, 0),
        reverse=True
    )

    highest = ranked[0]
    extras = [q for q in ranked[1:] if q != highest]

    return highest, extras


def extract_resume_data(text):

    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return {
            "full_name": "Test User",
            "gender": "Male",
            "age": 25,
            "city": "Delhi",

            "subjects": ["Mathematics"],
            "grade_levels": ["Class 11-12"],
            "languages": ["English"],

            "college_type": None,
            "college": ["University of Delhi"],
            "education_history": ["B.Sc Mathematics, University of Delhi"],

            "current_institution": "ABC School",
            "current_designation": "PGT Mathematics",
            "previous_institutions": ["XYZ School"],

            "experience_years": 3,
            "preferred_job_type": "Full-time",

            "email": "test@example.com",
            "phone": "+919999999999",

            "qualification": "M.Sc",
            "extra_qualifications": [],

            "extraction_failed": False
        }

    short_text = text[:8000]
    lower_text = text.lower()

    if (
        "offer letter" in lower_text
        or "appointment as" in lower_text
    ):
        return {
            "extraction_failed": True,
            "error": "Document appears to be an offer letter, not a resume"
        }

    prompt = f"""
You are an expert teacher resume parser. Extract structured data from the resume text below.

Return ONLY valid JSON, matching this EXACT schema. No explanations, no markdown, no extra keys.

{{
    "full_name": null,
    "gender": null,
    "age": null,
    "city": null,
    "subjects": [],
    "grade_levels": [],
    "languages": [],
    "skills": [],
    "college_type": null,
    "college": [],
    "education_history": [],
    "current_institution": null,
    "current_designation": null,
    "previous_institutions": [],
    "experience_years": null,
}}

CRITICAL RULES — follow these exactly:

- "age" must be a plain integer (e.g. 28), representing the person's current age in years.
  NEVER put a date of birth, a year, or any text in "age". If the resume doesn't state an
  actual age (a number of years), leave "age" as null. If DOB or birth year is present,
calculate current age in years. Return a plain integer age.
Examples:
02-09-2002 -> 23
23/02/1974 -> 52
If no DOB or age information exists, return null.
- "gender" — if not explicitly stated in the resume, INFER it from the person's first name
  using common Indian naming conventions (e.g. "Priya" -> Female, "Rajesh" -> Male). Return
  "Male" or "Female". Only leave this null if the name is genuinely ambiguous or unavailable
  (e.g. only initials given, like "A. Kumar").
- "college_type" must be EXACTLY one of: "IIT", "NIT", or null.
  Set it to "IIT" only if the person studied (any degree) at any Indian Institute of
  Technology. Set it to "NIT" only if they studied at any National Institute of Technology.
  If neither applies, leave it null — do NOT put any other institution name, degree type,
  or descriptive text in this field.
- skills must be a list of technical or professional skills.
    Examples:
    ["MS-CIT", "Tally Prime", "C++", "R", "LaTeX"]
- college must contain ONLY the institutions
associated with the candidate's highest
educational qualifications.
Maximum 3 institutions.
Do NOT include:
- SSC schools
- HSC schools
- coaching centres
- certification institutes
- every institution from education history
Examples:
GOOD:
["IIT Delhi"]
GOOD:
["Yashwantrao Chavan University",
 "North Maharashtra University"]
BAD:
["School",
 "College",
 "SSC Board",
 "Coaching Centre",
 "University",
 "Training Institute"] — plain strings like
  "Indian Institute of Technology, Bombay" or "University of Delhi". Do NOT put degree
  names, years, or percentages in this field (e.g. "B.Sc (2021)" is WRONG — that belongs
  in education_history, not here).
- city must be a single city/town name only.

- Do NOT guess, infer, correct, or construct city names.
- current_designation should be the person's most recent
  job title/designation.

- Common examples:
  Faculty
  Biology Faculty
  Mathematics Faculty
  Physics Faculty
  Chemistry Faculty
  Teacher
  Lecturer
  Professor
  Assistant Professor
  Principal
  Coordinator
  Trainer

- If the resume mentions "Faculty",
  "Teaching Assistant",
  "Professor",
  "Lecturer",
  etc., do not leave current_designation blank.

- Return null only when no designation can be identified.
- Do NOT return locality names, colony names,
  streets, areas, villages, districts,
  addresses, landmarks, or combinations of locations.

- If multiple locations are present,
  return only the actual city/town name.

Examples:

GOOD:
"Mumbai"
"Dhule"
"Buldana"
"Ichalkaranji"
"Delhi"

BAD:
"Shivaji Nagar Korochi"
"Sector 62 Noida"
"Near Bus Stand"
"Taluka Hatkanangale"
"Korochi, Kolhapur"
"Village XYZ"

- If a clear city/town name cannot be determined,
  return null.
- education_history should include only graduation,
  post-graduation, diploma, doctorate,
  professional qualifications, certifications,
  and higher education.
- previous_institutions must contain only places
  where the candidate WORKED.
- If an institution appears in education_history,
  it should NOT appear in previous_institutions
  unless the resume explicitly states that the
  candidate worked there.
- Do NOT include:
  colleges
  universities
  schools attended as a student
  training institutes
  certification institutes

- Include only organizations where the candidate
  was employed, taught, trained students,
  or held a professional role.

Examples:

GOOD:
Delhi Public School
Aakash Institute
FIITJEE
Allen Career Institute

BAD:
IIT Delhi
University of Delhi
North Maharashtra University
Jai Hind College
Canossa Convent High School
- Exclude SSC, HSC, Class 10,
  Class 12, Secondary Education,
  Higher Secondary Education entries.
- "education_history" must be a list of plain strings, one per degree/qualification,
  each combining degree + institution + year into ONE readable string, e.g.
  "M.Sc. Chemistry, XYZ University, 2020, 75%". Do not return nested objects.
- "subjects" must list the SPECIFIC academic subjects the person teaches or is qualified
  to teach (e.g. "Physics", "Organic Chemistry", "Mathematics") — not vague skills, not
  soft skills, not generic terms like "Teaching" or "Education". If the resume doesn't
  clearly state a teaching subject, leave the list empty rather than guessing.
- "grade_levels" must list the SPECIFIC class/grade ranges or exam levels the person
  teaches (e.g. "Class 9-10", "Class 11-12", "JEE Main & Advanced", "NEET") — not vague
  descriptions. Leave empty if not clearly stated.
- "experience_years" must be a plain integer — the person's TOTAL years of relevant
  teaching/professional work experience. Calculate this by looking at the work history
  section: if date ranges are given (e.g. "2019 - Present", "2015 - 2018"), compute the
  total span of relevant experience. If the resume explicitly states a number of years
  of experience, use that instead. If neither can be determined confidently, leave null
  — do NOT guess a number.
- "current_institution" must be the name of the school/institution the person CURRENTLY
  works at (their most recent employer), not a college.
- "current_designation" must be the person's CURRENT or MOST RECENT job title only
  (e.g. "Senior Physics Faculty", "PGT Chemistry Teacher") — not a list, not a summary
  of their whole career, just the one current/latest title.
- "previous_institutions" must be a list of plain strings — the names of EMPLOYERS
  (schools/institutions the person has WORKED at, not studied at) prior to their current
  one, e.g. ["Delhi Public School, Noida", "Ryan International School"]. Do NOT include
  the current institution in this list, and do NOT include colleges/universities here —
  those belong in "college". Leave empty if the resume shows no prior employer or only
  one job overall.
- Do not guess or hallucinate any value except gender as instructed above.
  If something else isn't clearly stated, use null (or an empty list for list fields).
- languages must be a list of plain strings.
- Return valid JSON only — nothing before or after it.

Resume:

{short_text}
"""

    parsed = {}
    extraction_failed = False

    max_attempts = 4
    base_delay = 2 

    for attempt in range(max_attempts):

        try:

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            content = (
                response.choices[0]
                .message.content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            json_match = re.search(r"\{.*\}", content, re.DOTALL)

            if json_match:
                content = json_match.group(0)

            parsed = json.loads(content)

            if isinstance(parsed, list):

                if len(parsed) > 0:
                    parsed = parsed[0]
                else:
                    parsed = {}

            break

        except Exception as e:

            print(f"[OpenAI] Error on attempt: {type(e).__name__}: {e}")

            err_str = str(e)

            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            is_parse_error = isinstance(e, json.JSONDecodeError)

            if (is_rate_limit or is_parse_error) and attempt < max_attempts - 1:

                retry_after = None

                match = re.search(
                    r"try again in ([\d.]+)s",
                    err_str,
                    re.IGNORECASE
                )

                if match:
                    retry_after = float(match.group(1))

                delay = retry_after if retry_after else (base_delay * (2 ** attempt))
                delay += random.uniform(0, 1)

                time.sleep(delay)
                continue

            else:
                extraction_failed = True
                break

    else:
        extraction_failed = True

    qualification_list = extract_qualifications(text)

    highest_qualification, extra_qualification_list = split_qualifications(qualification_list)

    parsed["email"] = extract_email(text)
    parsed["phone"] = extract_phone(text)

    llm_experience = sanitize_experience(parsed.get("experience_years"))

    parsed["experience_years"] = llm_experience if llm_experience is not None else extract_experience(text)

    parsed["qualification"] = highest_qualification
    parsed["extra_qualifications"] = extra_qualification_list

    parsed.setdefault("full_name", None)
    parsed.setdefault("gender", None)
    parsed.setdefault("age", None)
    parsed.setdefault("city", None)

    parsed.setdefault("subjects", [])
    parsed.setdefault("grade_levels", [])
    parsed.setdefault("languages", [])
    parsed.setdefault("skills", [])

    parsed.setdefault("college_type", None)

    parsed.setdefault("college", [])
    parsed.setdefault("education_history", [])

    parsed.setdefault("current_institution", None)
    parsed.setdefault("current_designation", None)
    parsed.setdefault("previous_institutions", [])
    parsed["age"] = sanitize_age(parsed.get("age"))
    parsed["experience_years"] = sanitize_experience(parsed.get("experience_years"))
    parsed["college_type"] = sanitize_college_type(parsed.get("college_type"))

    parsed["phone"] = sanitize_phone(parsed.get("phone"))

    parsed["college"] = flatten_list_field(parsed.get("college"))
    parsed["education_history"] = flatten_list_field(parsed.get("education_history"))
    parsed["subjects"] = flatten_list_field(parsed.get("subjects"))
    parsed["grade_levels"] = flatten_list_field(parsed.get("grade_levels"))
    parsed["languages"] = flatten_list_field(parsed.get("languages"))
    parsed["skills"] = flatten_list_field(parsed.get("skills"))
    parsed["previous_institutions"] = flatten_list_field(parsed.get("previous_institutions"))

    parsed.pop("current_salary", None)
    parsed.pop("expected_salary", None)

    parsed["extraction_failed"] = extraction_failed

    return parsed