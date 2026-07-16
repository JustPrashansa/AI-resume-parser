import os
import gspread
import streamlit as st

from google.oauth2 import service_account

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_NAME = "Resume_Master_DB"


def _get_credentials():
    """
    Works both locally (a JSON key file on disk) and on Streamlit
    Community Cloud (the JSON content pasted into Secrets — no file
    needed). Streamlit secrets take priority when present.
    """

    try:
        if "gcp_service_account" in st.secrets:
            return service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=SCOPES
            )
    except Exception:
        pass

    service_account_file = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "symbolic-axe-502107-r1-2ff6db019a1f.json"
    )

    return service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=SCOPES
    )

_sheet_cache = {"sheet": None}


def get_sheet(force_refresh=False):

    if _sheet_cache["sheet"] is not None and not force_refresh:
        return _sheet_cache["sheet"]

    creds = _get_credentials()

    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1

    _sheet_cache["sheet"] = sheet

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
        ", ".join(candidate.get("subjects") or []),
        ", ".join(candidate.get("grade_levels") or []),
        ", ".join(candidate.get("languages") or []),
        candidate.get("qualification"),
        ", ".join(candidate.get("extra_qualifications") or []),
        candidate.get("college_type"),
        ", ".join(candidate.get("college") or []),
        ", ".join(candidate.get("education_history") or []),
        candidate.get("experience_years"),
        candidate.get("current_institution"),
        candidate.get("current_designation"),
        ", ".join(candidate.get("previous_institutions") or []),
        ", ".join(candidate.get("skills") or []),
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