from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import io
import os
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Google Docs/Sheets/Slides can't be downloaded via get_media() directly
GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps"

# googleapiclient's service object (and the httplib2/Http connection under
# it) is NOT thread-safe. Sharing one instance across threads causes silent
# corruption or crashes. So each thread gets its OWN service, built once
# and cached for that thread only.
_thread_local = threading.local()


def get_drive_service():

    if not hasattr(_thread_local, "service"):

        creds = service_account.Credentials.from_service_account_file(
            "symbolic-axe-502107-r1-2ff6db019a1f.json",
            scopes=SCOPES
        )

        _thread_local.service = build(
            "drive",
            "v3",
            credentials=creds
        )

    return _thread_local.service


def extract_folder_id(folder_url):

    if "/folders/" in folder_url:
        return folder_url.split("/folders/")[1].split("?")[0]

    raise ValueError("Invalid Google Drive folder link")


def get_files_from_folder(folder_url, service=None):

    if service is None:
        service = get_drive_service()

    folder_id = extract_folder_id(folder_url)

    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType)"
    ).execute()

    all_files = results.get("files", [])

    # Split out real binary files vs native Google Docs files.
    # Native Google Docs (mimeType starts with application/vnd.google-apps)
    # need export(), not get_media(), and are usually not resumes anyway.
    downloadable = [
        f for f in all_files
        if not f.get("mimeType", "").startswith(GOOGLE_NATIVE_MIME_PREFIX)
    ]

    skipped = [
        f for f in all_files
        if f.get("mimeType", "").startswith(GOOGLE_NATIVE_MIME_PREFIX)
    ]

    return downloadable, skipped


def download_file(file_id, file_name, output_dir):

    # Each thread fetches its OWN service instance (thread-local, cached).
    service = get_drive_service()

    request = service.files().get_media(
        fileId=file_id
    )

    file_path = os.path.join(
        output_dir,
        file_name
    )

    fh = io.FileIO(
        file_path,
        "wb"
    )

    downloader = MediaIoBaseDownload(
        fh,
        request
    )

    done = False

    while not done:
        status, done = downloader.next_chunk()

    fh.close()

    return file_path


def download_all_files(files, output_dir, max_workers=5, progress_callback=None):
    """
    Downloads all files in parallel. Each thread uses its own cached
    Drive service (thread-local) instead of rebuilding auth per file,
    and without unsafely sharing one connection across threads.

    Returns (successful_paths, failed_files) where failed_files is a
    list of dicts: {"name": ..., "error": ...}
    """

    successful = []
    failed = []
    completed = 0
    total = len(files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_file = {
            executor.submit(
                download_file,
                f["id"],
                f["name"],
                output_dir
            ): f
            for f in files
        }

        for future in as_completed(future_to_file):

            file_info = future_to_file[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

            try:
                path = future.result()
                successful.append(path)

            except Exception as e:
                failed.append({
                    "name": file_info["name"],
                    "error": str(e)
                })

    return successful, failed