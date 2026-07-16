from sheets_utils import (
    append_candidate,
    get_existing_hashes,
    get_existing_contacts
)
from sheets_utils import get_sheet
import streamlit as st
import pandas as pd
import os
import shutil
import json
import hashlib
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

from drive_utils import (
    get_files_from_folder,
    download_file,
)

from extractor import (
    read_pdf,
    read_docx,
    extract_resume_data
)

st.set_page_config(
    page_title="Teacher Resume Parser",
    layout="wide"
)

st.title("Teacher Resume Parser")


def get_secret(name, default=None):
    """
    Works both locally (.env via os.environ, loaded by extractor.py's
    load_dotenv()) and on Streamlit Community Cloud (st.secrets, set in
    the app's Settings -> Secrets). Streamlit secrets take priority so a
    deployed app doesn't need a .env file at all.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it to your .env file locally, or "
        "to this app's Secrets if it's deployed on Streamlit Cloud."
    )
    st.stop()
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

drive_link = st.text_input(
    "Paste Google Drive Folder Link"
)

OPENAI_WORKERS = int(get_secret("OPENAI_WORKERS", 5))
DOWNLOAD_WORKERS = 5

def hash_text(text):
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def process_resume(file_info, output_dir, existing_hashes):
    file_id = file_info["id"]
    file_name = file_info["name"]

    try:

        file_path = os.path.join(output_dir, file_name)

        if file_name.lower().endswith(".pdf"):
            text = read_pdf(file_path)
        elif file_name.lower().endswith(".docx"):
            text = read_docx(file_path)
        else:
            return None

        content_hash = hash_text(text)

        if content_hash in existing_hashes:
            return {
                "duplicate_of_content": True,
                "resume_file_name": file_name,
                "content_hash": content_hash
            }

        data = extract_resume_data(text)
        data["content_hash"] = content_hash
        data["file_id"] = file_id
        data["resume_link"] = f"https://drive.google.com/file/d/{file_id}/view"
        data["resume_file_name"] = file_name
        data["duplicate_of_content"] = False
        return data

    except Exception as e:

        return {"error": f"{file_name}: {str(e)}"}


if st.button("Process Resumes"):

    if not drive_link:
        st.error("Please enter a Google Drive folder link.")
        st.stop()

    output_dir = "temp_resumes"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    st.write("Fetching files from Google Drive...")

    files, skipped_files = get_files_from_folder(drive_link)

    if skipped_files:
        skipped_names = ", ".join(f["name"] for f in skipped_files)
        st.warning(
            f"Skipped {len(skipped_files)} native Google Docs/Sheets/Slides "
            f"(not downloadable as-is): {skipped_names}"
        )

    if not files:
        st.error("No downloadable files found.")
        st.stop()

    st.write(f"Found {len(files)} files")

    existing_hashes = get_existing_hashes()
    existing_emails, existing_phones = get_existing_contacts()

    st.info(
        f"{len(existing_hashes)} resumes already in the master sheet by content — "
        f"{len(files)} files in this folder will be checked against them now "
        f"(downloads happen regardless; duplicates are skipped before any OpenAI call)."
    )

    download_progress = st.progress(0)
    process_progress = st.progress(0)

    st.write("Downloading + processing resumes...")

    results = []
    download_failures = []
    total = len(files)
    downloaded_count = 0
    processed_count = 0

    download_executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    groq_executor = ThreadPoolExecutor(max_workers=OPENAI_WORKERS)

    download_futures = {
        download_executor.submit(
            download_file, f["id"], f["name"], output_dir
        ): f
        for f in files
    } if files else {}

    process_futures = {}

    for future in as_completed(download_futures):

        file_info = download_futures[future]
        downloaded_count += 1
        if total:
            download_progress.progress(downloaded_count / total)

        try:
            future.result()
            process_future = groq_executor.submit(
                process_resume,
                file_info,
                output_dir,
                existing_hashes
            )
            process_futures[process_future] = file_info

        except Exception as e:
            download_failures.append({"name": file_info["name"], "error": str(e)})
            st.error(f"Download failed: {file_info['name']} ({str(e)})")

    for future in as_completed(process_futures):

        file_info = process_futures[future]
        processed_count += 1
        if process_futures:
            process_progress.progress(processed_count / len(process_futures))

        result = future.result()

        if result:
            if "error" in result:
                st.error(result["error"])
            else:

                if (
                    not result.get("extraction_failed")
                    and not result.get("duplicate_of_content")
                ):

                    email = str(result.get("email", "")).strip().lower()
                    phone = str(result.get("phone", "")).strip()

                    if (email and email in existing_emails) or (phone and phone in existing_phones):

                        st.warning(
                            f"Skipping existing candidate: {result.get('full_name')}"
                        )

                        result["skipped_existing_contact"] = True
                        results.append(result)
                        continue

                    append_candidate(result)
                    result["written_to_sheet"] = True

                    if result.get("content_hash"):
                        existing_hashes.add(str(result["content_hash"]))

                    if email:
                        existing_emails.add(email)
                    if phone:
                        existing_phones.add(phone)

                results.append(result)

    download_executor.shutdown(wait=True)
    groq_executor.shutdown(wait=True)

    duplicate_content_count = sum(1 for r in results if r.get("duplicate_of_content"))
    skipped_existing_count = sum(1 for r in results if r.get("skipped_existing_contact"))

    failed_extraction = [r for r in results if r.get("extraction_failed")]

    unique_results = [
        r for r in results
        if r.get("written_to_sheet")
    ]

    st.write(
        f"Done. {len(results)} files processed "
        f"({duplicate_content_count} duplicate content, "
        f"{skipped_existing_count} matched an existing candidate, "
        f"{len(failed_extraction)} failed extraction and were NOT written to the sheet). "
        f"{len(unique_results)} new candidates actually added to the sheet this run."
    )

    if failed_extraction:

        failed_names = ", ".join(
            r.get("resume_file_name", "unknown") for r in failed_extraction
        )

        st.error(
            f"⚠️ {len(failed_extraction)} resumes could not be extracted "
            f"(errored after all retries — only email/phone were regex-extracted "
            f"for these, everything else is blank). "
            f"Re-run later to retry them: {failed_names}"
        )

    if unique_results:

        df = pd.DataFrame(unique_results)

        list_columns = [
            "subjects",
            "grade_levels",
            "languages",
            "skills",
            "extra_qualifications",
            "college",
            "education_history",
            "previous_institutions",
        ]

        for col in list_columns:

            if col in df.columns:

                df[col] = df[col].apply(
                    lambda x:
                    "; ".join(
                        item if isinstance(item, str)
                        else json.dumps(item) if isinstance(item, dict)
                        else str(item)
                        for item in x
                    )
                    if isinstance(x, list)
                    else x
                )

        for col in ["age", "experience_years"]:

            if col in df.columns:
                df[col] = pd.array(df[col], dtype="Int64")

        if "email" in df.columns:
            df = df.drop_duplicates(subset=["email"], keep="first")

        if "phone" in df.columns:
            df = df.drop_duplicates(subset=["phone"], keep="first")

        desired_order = [
            "full_name",
            "email",
            "phone",
            "gender",
            "age",
            "city",

            "subjects",
            "grade_levels",
            "languages",

            "qualification",
            "extra_qualifications",

            "college_type",
            "college",
            "education_history",

            "experience_years",

            "current_institution",
            "current_designation",
            "previous_institutions",
            "skills",
            "resume_link"
        ]

        existing_columns = [col for col in desired_order if col in df.columns]

        df = df[existing_columns]

        excel_file = "candidates.xlsx"

        df.to_excel(excel_file, index=False)
        st.success(f"Processed {len(df)} unique candidates")
        st.dataframe(df, width="stretch")

        with open(excel_file, "rb") as f:

            st.download_button(
                label="Download Excel",
                data=f,
                file_name="candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        if duplicate_content_count > 0:
            st.success(
                f"All {duplicate_content_count} resumes were already processed before."
            )
        else:
            st.warning("No candidate data extracted.")