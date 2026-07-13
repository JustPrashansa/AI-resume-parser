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
    download_file
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

drive_link = st.text_input(
    "Paste Google Drive Folder Link"
)

# Switched to llama-3.3-70b-versatile for much better extraction accuracy
# (see extractor.py). It's slower and Groq's free tier RPM limit is lower
# for 70b models than for the 8b-instant model, so concurrency is reduced
# here to avoid constant 429 retries. Bump this up if you're on a paid tier.
GROQ_WORKERS = 2
DOWNLOAD_WORKERS = 5

# Cache #1: already-processed Drive file IDs -> result.
# Lets us skip download + Groq entirely for files we've seen before
# (same file, re-scanned on a later run).
CACHE_FILE = "processed_cache.json"

# Cache #2: content hash -> result.
# Catches TRUE duplicates — e.g. same CV re-uploaded as a new file with
# a new Drive ID (or a different filename). We hash the extracted resume
# text (free, local step) and skip the Groq call if we've seen that exact
# content before, reusing the old parsed result instead.
CONTENT_CACHE_FILE = "content_hash_cache.json"

# Content cache is read/written from multiple threads during parallel
# processing, so it needs a lock to stay safe.
content_cache_lock = threading.Lock()


def load_json_cache(path):

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        # Corrupted cache shouldn't block the whole run — start fresh.
        return {}


def save_json_cache(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache():
    return load_json_cache(CACHE_FILE)


def save_cache(cache):
    save_json_cache(CACHE_FILE, cache)


def hash_text(text):
    # Normalize whitespace so trivial formatting differences (extra
    # spaces/newlines from re-saving a PDF) don't cause a false "new" hit.
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def process_resume(file_info, output_dir, content_cache):

    file_id = file_info["id"]
    file_name = file_info["name"]

    try:

        file_path = os.path.join(
            output_dir,
            file_name
        )

        if file_name.lower().endswith(".pdf"):

            text = read_pdf(file_path)

        elif file_name.lower().endswith(".docx"):

            text = read_docx(file_path)

        else:

            return None

        content_hash = hash_text(text)

        # Check if we've already sent this EXACT content to Groq before
        # (e.g. same CV re-uploaded under a different file name/ID).
        # If so, reuse that result instead of paying for another API call.
        with content_cache_lock:
            cached_result = content_cache.get(content_hash)

        if cached_result is not None:

            data = dict(cached_result)  # copy, don't mutate the cached one
            data["resume_link"] = (
                f"https://drive.google.com/file/d/{file_id}/view"
            )
            data["resume_file_name"] = file_name
            data["duplicate_of_content"] = True

            return data

        # New content — actually call Groq.
        data = extract_resume_data(text)

        data["resume_link"] = (
            f"https://drive.google.com/file/d/{file_id}/view"
        )

        data["resume_file_name"] = file_name
        data["duplicate_of_content"] = False

        # Only cache genuinely successful extractions. If Groq failed for
        # this resume, caching it would make it look "done" forever and
        # the "rerun later to retry" message would be a lie — a future
        # run needs to actually retry it, not skip it as already-known.
        if not data.get("extraction_failed"):
            with content_cache_lock:
                content_cache[content_hash] = data

        return data

    except Exception as e:

        return {
            "error": f"{file_name}: {str(e)}"
        }


if st.button("Process Resumes"):

    if not drive_link:

        st.error(
            "Please enter a Google Drive folder link."
        )

        st.stop()

    output_dir = "temp_resumes"

    if os.path.exists(output_dir):

        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    st.write(
        "Fetching files from Google Drive..."
    )

    files, skipped_files = get_files_from_folder(
        drive_link
    )

    if skipped_files:

        skipped_names = ", ".join(f["name"] for f in skipped_files)

        st.warning(
            f"Skipped {len(skipped_files)} native Google Docs/Sheets/Slides "
            f"(not downloadable as-is): {skipped_names}"
        )

    if not files:

        st.error(
            "No downloadable files found."
        )

        st.stop()

    st.write(
        f"Found {len(files)} files"
    )

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

    # PIPELINE: download and process concurrently. As soon as a file
    # finishes downloading, it's submitted for Groq processing right away
    # — we don't wait for all 25 downloads to finish before processing
    # starts. This is the main speed win, since the Groq call is the
    # slowest step and was previously sitting idle during downloads.

    download_progress = st.progress(0)
    process_progress = st.progress(0)

    st.write("Downloading + processing resumes...")

    results = []
    download_failures = []
    total = len(files)
    downloaded_count = 0
    processed_count = 0

    download_executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    groq_executor = ThreadPoolExecutor(max_workers=GROQ_WORKERS)

    download_futures = {
        download_executor.submit(
            download_file,
            f["id"],
            f["name"],
            output_dir
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

            # File is on disk now — kick off processing immediately,
            # don't wait for other downloads.
            process_future = groq_executor.submit(
                process_resume,
                file_info,
                output_dir,
                content_cache
            )
            process_futures[process_future] = file_info

        except Exception as e:
            download_failures.append({
                "name": file_info["name"],
                "error": str(e)
            })
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

                # Save into the cache keyed by Drive file ID, so tomorrow's
                # run knows this one's already done and skips it — UNLESS
                # extraction failed, in which case we deliberately do NOT
                # cache it, so a future run actually retries it instead of
                # treating a failed/blank result as permanently "done".
                if not result.get("extraction_failed"):
                    cache[file_info["id"]] = result

    download_executor.shutdown(wait=True)
    groq_executor.shutdown(wait=True)

    save_cache(cache)
    save_json_cache(CONTENT_CACHE_FILE, content_cache)

    duplicate_content_count = sum(
        1 for r in results if r.get("duplicate_of_content")
    )

    failed_extraction = [
        r for r in results if r.get("extraction_failed")
    ]

    st.write(
        f"Done. {len(results)} new files handled "
        f"({duplicate_content_count} were duplicate content — skipped Groq, reused cached result). "
        f"{len(cache)} total candidates in cache."
    )

    if failed_extraction:

        failed_names = ", ".join(
            r.get("resume_file_name", "unknown") for r in failed_extraction
        )

        st.error(
            f"⚠️ {len(failed_extraction)} resumes could not be extracted by Groq "
            f"(rate-limited or errored after all retries — only email/phone were "
            f"regex-extracted for these, everything else is blank). "
            f"Re-run later to retry them: {failed_names}"
        )

    # BUILD DATAFRAME from the FULL cache (old + new), not just today's
    # new results — this is what makes the Excel always contain
    # everyone ever processed, without re-calling Groq on repeats.

    all_results = list(cache.values())

    if all_results:

        df = pd.DataFrame(
            all_results
        )

        list_columns = [

            "subjects",
            "grade_levels",
            "languages",
            "extra_qualifications",
            "college",
            "education_history"

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

                    if isinstance(
                        x,
                        list
                    )

                    else x

                )

        # 'age' (and a few other fields) can come back as int, string, or
        # None depending on what the LLM returns per resume. Mixed types
        # in one column break PyArrow's dataframe rendering in Streamlit.
        # Force these to plain strings so the column type is consistent.
        for col in ["age", "experience_years"]:

            if col in df.columns:

                df[col] = df[col].apply(
                    lambda x: str(x) if x is not None else None
                )

        # REMOVE DUPLICATES

        if "email" in df.columns:

            df = df.drop_duplicates(

                subset=["email"],
                keep="first"

            )

        if "phone" in df.columns:

            df = df.drop_duplicates(

                subset=["phone"],
                keep="first"

            )

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

            "preferred_job_type",

            "resume_link"

        ]

        existing_columns = [

            col

            for col in desired_order

            if col in df.columns

        ]

        df = df[
            existing_columns
        ]

        excel_file = (
            "candidates.xlsx"
        )

        df.to_excel(

            excel_file,

            index=False

        )

        st.success(

            f"Processed {len(df)} unique candidates"

        )

        st.dataframe(

            df,

            width="stretch"

        )

        with open(

            excel_file,

            "rb"

        ) as f:

            st.download_button(

                label="Download Excel",

                data=f,

                file_name="candidates.xlsx",

                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    else:

        st.warning(
            "No candidate data extracted."
        )