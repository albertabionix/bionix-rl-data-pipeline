"""Download raw recordings from the Alberta Bionix Shared Drive via BionixDB.

Files are saved unchanged into a local folder (the cache), which the rest of
the pipeline then reads like any other folder of CSVs. Files already in the
cache with the same size are not downloaded again.

Only real CSV files are fetched. The Drive's dataset folders also contain a
few Google Sheets copies of recordings (e.g. ``emg-p006-seated-01`` opened in
Sheets); BionixDB's own ``download()`` fails on those, so this lists files
itself and skips anything that isn't ``text/csv``.

Needs the optional dependency: ``pip install -e '.[drive]'``.
"""
from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import MODALITIES, parse_key


def _list_csvs(db, folder_id: str, drive_id: str) -> list[dict]:
    files, token = [], None
    while True:
        resp = db.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'",
            corpora="drive", driveId=drive_id, includeItemsFromAllDrives=True,
            supportsAllDrives=True, pageSize=1000, pageToken=token,
            fields="nextPageToken, files(id, name, size)",
        ).execute()
        files += resp.get("files", [])
        token = resp.get("nextPageToken")
        if not token:
            return files


def _matches(name: str, pid, action, exercise) -> bool:
    if not name.endswith(".csv"):
        return False
    key = parse_key(name[:-4].split("-", 1)[1]) if "-" in name else {}
    if pid is not None and key.get("pid") != (f"p{int(pid):03d}" if str(pid).isdigit() else pid):
        return False
    if action is not None and key.get("action") != action:
        return False
    if exercise is not None and key.get("exercise") != exercise:
        return False
    return True


def fetch(cache: str | Path, pid=None, action=None, exercise=None,
          modalities=MODALITIES, token_file="token.json", workers: int = 8) -> list[Path]:
    """Download matching recordings into ``cache``; returns all matching local paths.

    Opens a browser for Google login the first time (BionixDB caches the token
    in ``token_file``).
    """
    from bionix_db import BionixDB
    from bionix_db.bionixdb import BIONIX_DRIVE_ID, MODALITY_FOLDERS
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    db = BionixDB(token_file=str(token_file))

    wanted = []
    for m in modalities:
        wanted += [f for f in _list_csvs(db, MODALITY_FOLDERS[m], BIONIX_DRIVE_ID)
                   if _matches(f["name"], pid, action, exercise)]
    todo = [f for f in wanted
            if not (cache / f["name"]).exists()
            or (cache / f["name"]).stat().st_size != int(f.get("size", -1))]
    print(f"Drive: {len(wanted)} matching files, {len(todo)} to download into {cache}/")

    local = threading.local()

    def download(f) -> str | None:
        if not hasattr(local, "service"):  # the Drive client isn't thread-safe
            local.service = build("drive", "v3", credentials=db.creds, cache_discovery=False)
        error = None
        for attempt in range(4):
            try:
                buf = io.BytesIO()
                req = local.service.files().get_media(fileId=f["id"], supportsAllDrives=True)
                dl = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                tmp = cache / (f["name"] + ".part")
                tmp.write_bytes(buf.getvalue())
                tmp.replace(cache / f["name"])  # never leave a half-written CSV behind
                return None
            except Exception as e:  # network hiccups: retry with backoff
                error = e
                time.sleep(2 ** attempt)
        return f"{f['name']}: {error}"

    with ThreadPoolExecutor(workers) as pool:
        errors = [e for e in pool.map(download, todo) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed, e.g. {errors[:3]}")
    return sorted(cache / f["name"] for f in wanted)
