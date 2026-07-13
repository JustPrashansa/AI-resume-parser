import json
import re
import time
import random
import pdfplumber
from docx import Document
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# llama-3.1-8b-instant is fast but genuinely inaccurate for structured
# extraction — it confuses dates of birth with age, and returns wildly
# inconsistent JSON shapes (education_history as strings one time, nested
# dicts the next). llama-3.3-70b-versatile is much more reliable at
# following the schema, BUT Groq's free tier gives it a much stricter
# rate limit than the 8B model. If 70B keeps hitting 429s and exhausts
# retries, we fall back to 8B rather than silently returning nothing —
# a slightly-less-accurate result beats a blank row.
GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"


def sanitize_age(value):
    """
    The LLM sometimes returns a birthdate, a range, or other garbage in
    the 'age' field instead of a plain integer. Only accept something
    that's actually a plausible human age; everything else becomes None
    rather than polluting the data with junk like "January 02, 1987".
    """

    if value is None:
        return None

    try:
        age_int = int(str(value).strip())

        if 15 <= age_int <= 80:
            return age_int

    except (ValueError, TypeError):
        pass

    return None


def sanitize_number(value):
    """Coerce salary-like fields to a plain number, or None if unusable."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    # Strip common junk like "₹", commas, "LPA", "per month" etc.
    cleaned = re.sub(r"[^\d.]", "", str(value))

    if not cleaned:
        return None

    try:
        return float(cleaned) if "." in cleaned else int(cleaned)

    except ValueError:
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

    # Strip a leading country code (91) if present, so we're always
    # left with just the 10-digit number before re-adding +91.
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]

    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    if len(digits) == 10 and digits[0] in "6789":
        return f"+91{digits}"

    return None


def flatten_value(item):
    """
    The LLM is inconsistent about whether education_history/college
    entries are plain strings or nested objects with arbitrary keys.
    This forces everything into a single readable string so every
    record in the cache/Excel has the same shape, instead of some rows
    being strings and others being raw JSON dumps.
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


def read_pdf(file, char_limit=5000):
    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

            # extract_resume_data only ever uses the first 5000 chars,
            # so stop parsing further pages once we have enough —
            # this avoids wasting time on long multi-page resumes.
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

    matches = re.findall(
        r'(\d+)\+?\s*years',
        text,
        re.IGNORECASE
    )

    if matches:

        try:
            return max(
                int(x)
                for x in matches
            )

        except:
            pass

    return None


def extract_qualifications(text):

    qualifications = []

    keywords = [
        "B.Ed",
        "M.Ed",
        "D.El.Ed",
        "CTET",
        "TET",
        "NET",
        "UGC NET",
        "JRF",
        "PhD",
        "B.Sc",
        "M.Sc",
        "B.A",
        "M.A",
        "B.Com",
        "M.Com",
        "B.Tech",
        "M.Tech"
    ]

    lower_text = text.lower()

    for keyword in keywords:

        if keyword.lower() in lower_text:
            qualifications.append(keyword)

    return list(set(qualifications))


# Rough seniority ranking used to pick the TRUE highest qualification out
# of everything found, instead of picking an arbitrary one from a set
# (Python sets have no order, which is why "qualification" was previously
# showing basically a random match instead of the highest one).
QUALIFICATION_RANK = {
    "PhD": 100,
    "JRF": 95,
    "UGC NET": 90,
    "NET": 90,
    "M.Ed": 80,
    "M.Tech": 75,
    "M.Sc": 75,
    "M.A": 75,
    "M.Com": 75,
    "B.Ed": 60,
    "B.Tech": 55,
    "B.Sc": 55,
    "B.A": 55,
    "B.Com": 55,
    "D.El.Ed": 40,
    "CTET": 30,
    "TET": 30,
}


def split_qualifications(qualification_list):
    """
    Returns (highest_qualification, extra_qualifications) — the highest
    by seniority rank, and everything else found (excluding the highest).
    """

    if not qualification_list:
        return None, []

    ranked = sorted(
        qualification_list,
        key=lambda q: QUALIFICATION_RANK.get(q, 0),
        reverse=True
    )

    highest = ranked[0]
    extras = ranked[1:]

    return highest, extras


def extract_resume_data(text):

    # Increased from 5000 to give the model more context — teacher resumes
    # are often dense, and cutting off early was losing education/subject
    # details that came later in the document.
    short_text = text[:8000]

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

    "college_type": null,

    "college": [],
    "education_history": [],

    "current_institution": null,
    "current_designation": null,

    "experience_years": null,

    "preferred_job_type": null
}}

CRITICAL RULES — follow these exactly:

- "age" must be a plain integer (e.g. 28), representing the person's current age in years.
  NEVER put a date of birth, a year, or any text in "age". If the resume doesn't state an
  actual age (a number of years), leave "age" as null. Do NOT calculate age from a birth date.
- "gender" — if not explicitly stated in the resume, INFER it from the person's first name
  using common Indian naming conventions (e.g. "Priya" -> Female, "Rajesh" -> Male). Return
  "Male" or "Female". Only leave this null if the name is genuinely ambiguous or unavailable
  (e.g. only initials given, like "A. Kumar").
- "college_type" must be EXACTLY one of: "IIT", "NIT", or null.
  Set it to "IIT" only if the person studied (any degree) at any Indian Institute of
  Technology. Set it to "NIT" only if they studied at any National Institute of Technology.
  If neither applies, leave it null — do NOT put any other institution name, degree type,
  or descriptive text in this field.
- "college" must be a list of INSTITUTION/UNIVERSITY NAMES ONLY — plain strings like
  "Indian Institute of Technology, Bombay" or "University of Delhi". Do NOT put degree
  names, years, or percentages in this field (e.g. "B.Sc (2021)" is WRONG — that belongs
  in education_history, not here).
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
- "current_designation" must be the person's CURRENT or MOST RECENT job title only
  (e.g. "Senior Physics Faculty", "PGT Chemistry Teacher") — not a list, not a summary
  of their whole career, just the one current/latest title.
- "preferred_job_type" — only fill this if the resume EXPLICITLY states a job type
  preference (e.g. "Full-time", "Remote", "Part-time"). If it's not explicitly stated,
  leave it null. Do not infer this from context.
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
    base_delay = 2  # seconds, doubles each retry: 2, 4, 8, 16 (+ jitter)

    for model_name in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):

        got_result = False

        for attempt in range(max_attempts):

            try:

                response = client.chat.completions.create(
                    model=model_name,
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

                # Defensive: even with response_format=json_object, if the
                # model prefixes any stray text, pull out just the {...}
                # block rather than failing the whole parse on it.
                json_match = re.search(r"\{.*\}", content, re.DOTALL)

                if json_match:
                    content = json_match.group(0)

                parsed = json.loads(content)

                if isinstance(parsed, list):

                    if len(parsed) > 0:
                        parsed = parsed[0]
                    else:
                        parsed = {}

                got_result = True
                break

            except Exception as e:

                err_str = str(e)

                is_rate_limit = "429" in err_str
                is_parse_error = isinstance(e, json.JSONDecodeError)

                # Retry on rate limits AND on malformed/truncated JSON —
                # previously a JSON parse failure (the actual common cause
                # of blank rows) gave up instantly with no retry at all.
                if (is_rate_limit or is_parse_error) and attempt < max_attempts - 1:

                    # Exponential backoff with jitter instead of a flat 15s.
                    # Groq free tier often gives a "try again in Xs" hint —
                    # use it if present, otherwise fall back to backoff.
                    retry_after = None

                    match = re.search(r"try again in ([\d.]+)s", err_str, re.IGNORECASE)

                    if match:
                        retry_after = float(match.group(1))

                    delay = retry_after if retry_after else (base_delay * (2 ** attempt))
                    delay += random.uniform(0, 1)  # jitter to avoid thundering herd

                    time.sleep(delay)

                    continue

                else:

                    break

        if got_result:
            break

    else:
        # Both models exhausted every retry — this resume's LLM extraction
        # genuinely failed. Flag it instead of silently returning a blank
        # record, so it's visible in the final output and you know to
        # rerun it rather than assume it's just an empty resume.
        extraction_failed = True

    qualification_list = extract_qualifications(text)

    highest_qualification, extra_qualification_list = split_qualifications(qualification_list)

    parsed["email"] = extract_email(text)
    parsed["phone"] = extract_phone(text)

    # Prefer the LLM's calculated experience (it can reason about date
    # ranges in the work history). Only fall back to the crude regex
    # scan if the model couldn't determine it.
    llm_experience = parsed.get("experience_years")

    try:
        llm_experience = int(llm_experience) if llm_experience is not None else None
    except (ValueError, TypeError):
        llm_experience = None

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

    parsed.setdefault("college_type", None)

    parsed.setdefault("college", [])
    parsed.setdefault("education_history", [])

    parsed.setdefault("current_institution", None)
    parsed.setdefault("current_designation", None)

    parsed.setdefault("preferred_job_type", None)

    # Normalize whatever the model returned so every cached record has the
    # same, clean shape — this is what actually fixes garbage values like
    # a birthdate in "age" or nested dicts in "education_history", instead
    # of just hiding them at display time.
    parsed["age"] = sanitize_age(parsed.get("age"))
    parsed["college_type"] = sanitize_college_type(parsed.get("college_type"))

    # Normalize phone to ONE consistent format (+91XXXXXXXXXX) regardless
    # of whether it came from the regex fallback or however the resume
    # had it written (with/without +91, spaces, dashes, etc).
    parsed["phone"] = sanitize_phone(parsed.get("phone"))

    parsed["college"] = flatten_list_field(parsed.get("college"))
    parsed["education_history"] = flatten_list_field(parsed.get("education_history"))
    parsed["subjects"] = flatten_list_field(parsed.get("subjects"))
    parsed["grade_levels"] = flatten_list_field(parsed.get("grade_levels"))
    parsed["languages"] = flatten_list_field(parsed.get("languages"))

    # Salary fields are no longer collected — drop them if the model
    # returned them anyway (e.g. from an older cached prompt behavior).
    parsed.pop("current_salary", None)
    parsed.pop("expected_salary", None)

    parsed["extraction_failed"] = extraction_failed

    return parsed