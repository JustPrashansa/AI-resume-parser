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
    download_cache_from_drive,
    upload_cache_to_drive,
    get_drive_service
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
DRIVE_CACHE_FOLDER_ID = get_secret("DRIVE_CACHE_FOLDER_ID")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it to your .env file locally, or "
        "to this app's Secrets if it's deployed on Streamlit Cloud."
    )
    st.stop()
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

if not DRIVE_CACHE_FOLDER_ID:
    st.warning(
        "DRIVE_CACHE_FOLDER_ID is not set — caches will only be saved "
        "locally on this machine for this session, not persisted to "
        "Google Drive. Set DRIVE_CACHE_FOLDER_ID to a Drive folder ID "
        "(shared with your service account) to persist across machines."
    )

drive_link = st.text_input(
    "Paste Google Drive Folder Link"
)

OPENAI_WORKERS = int(get_secret("OPENAI_WORKERS", 5))
DOWNLOAD_WORKERS = 5

CACHE_FILE = "processed_cache.json"
CONTENT_CACHE_FILE = "content_hash_cache.json"

content_cache_lock = threading.Lock()
def load_json_cache(file_name):

    if DRIVE_CACHE_FOLDER_ID:
        try:
            text = download_cache_from_drive(DRIVE_CACHE_FOLDER_ID, file_name)
            if text:
                return json.loads(text)
            return {}
        except Exception as e:
            st.warning(f"Could not load {file_name} from Drive ({e}). Starting fresh.")
            return {}

    if not os.path.exists(file_name):
        return {}

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_cache(file_name, data):

    text = json.dumps(data, ensure_ascii=False, indent=2)

    if DRIVE_CACHE_FOLDER_ID:
        try:
            upload_cache_to_drive(DRIVE_CACHE_FOLDER_ID, file_name, text)
            return
        except Exception as e:
            st.error(f"Could not save {file_name} to Drive ({e}). Saving locally instead.")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)


def load_cache():
    return load_json_cache(CACHE_FILE)


def save_cache(cache):
    save_json_cache(CACHE_FILE, cache)


def hash_text(text):
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def process_resume(file_info, output_dir, content_cache):

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

        with content_cache_lock:
            cached_result = content_cache.get(content_hash)

        if cached_result is not None:

            data = dict(cached_result)
            data["resume_link"] = f"https://drive.google.com/file/d/{file_id}/view"
            data["resume_file_name"] = file_name
            data["duplicate_of_content"] = True

            return data

        data = extract_resume_data(text)

        data["resume_link"] = f"https://drive.google.com/file/d/{file_id}/view"
        data["resume_file_name"] = file_name
        data["duplicate_of_content"] = False

        if not data.get("extraction_failed"):
            with content_cache_lock:
                content_cache[content_hash] = data

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

    cache = load_cache()
    content_cache = load_json_cache(CONTENT_CACHE_FILE)

    already_processed = [f for f in files if f["id"] in cache]
    new_files = [f for f in files if f["id"] not in cache]

    st.info(
        f"{len(already_processed)} already processed before (skipping) — "
        f"{len(new_files)} new files to process now."
    )

    if not new_files:
        st.success("Nothing new to process. Using cached results only.")

    files = new_files

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
                process_resume, file_info, output_dir, content_cache
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
                results.append(result)

                if not result.get("extraction_failed"):
                    cache[file_info["id"]] = result

    download_executor.shutdown(wait=True)
    groq_executor.shutdown(wait=True)

    save_cache(cache)
    save_json_cache(CONTENT_CACHE_FILE, content_cache)

    duplicate_content_count = sum(1 for r in results if r.get("duplicate_of_content"))

    failed_extraction = [r for r in results if r.get("extraction_failed")]

    st.write(
        f"Done. {len(results)} new files handled "
        f"({duplicate_content_count} were duplicate content — skipped the model call, reused cached result). "
        f"{len(cache)} total candidates in cache."
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

    all_results = list(cache.values())

    if all_results:

        df = pd.DataFrame(all_results)

        list_columns = [
            "subjects",
            "grade_levels",
            "languages",
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

            "preferred_job_type",

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
        st.warning("No candidate data extracted.")