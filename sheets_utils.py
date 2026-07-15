import os
import gspread

from google.oauth2 import service_account

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "symbolic-axe-502107-r1-2ff6db019a1f.json"
)

SHEET_NAME = "Resume_Master_DB"


def get_sheet():

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1

    return sheet


def append_candidate(candidate):

    sheet = get_sheet()

    row = [
        candidate.get("file_id"),
        candidate.get("content_hash"),
        candidate.get("resume_file_name"),
        candidate.get("full_name"),
        candidate.get("email"),
        candidate.get("phone"),
        candidate.get("gender"),
        candidate.get("age"),
        candidate.get("city"),
        ", ".join(candidate.get("subjects", [])),
        ", ".join(candidate.get("grade_levels", [])),
        ", ".join(candidate.get("languages", [])),
        candidate.get("qualification"),
        ", ".join(candidate.get("extra_qualifications", [])),
        candidate.get("college_type"),
        ", ".join(candidate.get("college", [])),
        ", ".join(candidate.get("education_history", [])),
        candidate.get("experience_years"),
        candidate.get("current_institution"),
        candidate.get("current_designation"),
        ", ".join(candidate.get("previous_institutions", [])),
        candidate.get("preferred_job_type"),
        candidate.get("resume_link")
    ]

    sheet.append_row(row)


def get_existing_hashes():

    sheet = get_sheet()

    records = sheet.get_all_records()

    hashes = set()

    for row in records:

        h = row.get("content_hash")

        if h:
            hashes.add(str(h))

    return hashes
def get_existing_contacts():

    sheet = get_sheet()

    records = sheet.get_all_records()

    emails = set()
    phones = set()

    for row in records:

        if row.get("email"):
            emails.add(str(row["email"]).strip().lower())

        if row.get("phone"):
            phones.add(str(row["phone"]).strip())

    return emails, phones