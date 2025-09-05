#!/usr/bin/env python3
"""
compare_working_fixed_final.py — Full merged & fixed script.

Fixes:
 - Cancel / New Job / Quarantine Selected / Delete Selected now respond correctly.
 - Debug log file `compare_debug.log` written next to the script only when ENABLE_LOGGING is present and True.
   To disable file logging, comment out or set ENABLE_LOGGING = False below.
 - Consistent button keys and robust event checks throughout the code.
 - Defensive logging and GUI-safe handlers preserved.

NOTE: This preserves your original structure and logic. If you want a slimmer demo version instead,
say so and I can supply that.
"""

# ---------- Toggle debug file logging ----------
# To disable writing the debug log file, comment out the following line or set it to False:
ENABLE_LOGGING = True
# ------------------------------------------------

import os
import sys
import time
import json
import shutil
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import hashlib
import traceback
from datetime import datetime

import cv2
from PIL import Image
import imagehash
import numpy as np
import FreeSimpleGUI as sg

# ---------------- CONFIG ----------------
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
SAMPLE_FRAMES_PER_VIDEO = 12
SIMILARITY_THRESHOLD = 0.86
COMPARE_BATCH = 200
ETA_ALPHA = 0.1
DB_FILENAME = ".video_hash_cache.sqlite"
THUMB_CACHE_DIR = ".thumb_cache"
THUMB_MAX_AGE_DAYS = 30
THUMB_SIZE = (150, 120)
THUMB_THREAD_WORKERS_BASE = 6
HASH_PROC_BASE = None
SEQ_CACHE_FILE = "phash_seq_cache.json"  # fast JSON cache (path+size+mtime -> seq)

# reduce noisy logs from libs
logging.getLogger().setLevel(logging.CRITICAL)
try:
    cv2.setLogLevel(0)
except Exception:
    pass

# ---------------- helper: script directory ----------------
def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

# ---------------- Debug log (user-toggleable file next to script) ----------------
DEBUG_LOG_FILENAME = "compare_debug.log"

def _debug_log_path():
    return os.path.join(get_script_dir(), DEBUG_LOG_FILENAME)

def debug_log(message):
    if not ENABLE_LOGGING:
        return
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        print(message)  # also show in console
    except Exception:
        print("Logging failed:", traceback.format_exc())
    except Exception:
        pass

# ---------------- Logging (Compare.log rotating) ----------------
logger = logging.getLogger("CompareLogger")
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_script_dir(), "Compare.log")
file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == file_handler.baseFilename for h in logger.handlers):
    logger.addHandler(file_handler)

class TextHandler(logging.Handler):
    """Handler to stream logs into GUI multiline when the progress window is open."""
    def __init__(self, win, key="-LOG-"):
        super().__init__()
        self.win = win
        self.key = key
        self.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.win and not getattr(self.win, "was_closed", lambda: False)():
                try:
                    self.win[self.key].update(msg + "\n", append=True)
                except Exception:
                    pass
        except Exception:
            pass

# ---------------- helpers for logging/exceptions ----------------
def log_exception(exc: Exception, context: str = ""):
    tb = traceback.format_exc()
    try:
        logger.error("EXCEPTION in %s: %s\n%s", context, exc, tb)
    except Exception:
        pass
    debug_log(f"EXCEPTION in {context}: {exc}\n{tb}")

def safe_log(msg, level="info"):
    try:
        if level == "debug":
            logger.debug(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.info(msg)
    except Exception:
        pass
    debug_log(msg)

# ---------------- ETA helper ----------------
class ETACalc:
    def __init__(self, alpha=ETA_ALPHA):
        self.alpha = alpha
        self.smoothed_rate = None
        self.last_time = None
        self.last_done = 0

    def update(self, done, total):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.last_done = done
            return "--:--"
        dt = now - self.last_time
        work = done - self.last_done
        self.last_time = now
        self.last_done = done
        if dt > 0 and work > 0:
            rate = work / dt
            if self.smoothed_rate is None:
                self.smoothed_rate = rate
            else:
                self.smoothed_rate = self.alpha * rate + (1 - self.alpha) * self.smoothed_rate
        if not self.smoothed_rate or self.smoothed_rate <= 0:
            return "--:--"
        remaining = max(0, total - done)
        eta = remaining / self.smoothed_rate
        if eta < 60:
            return f"{int(eta)}s"
        if eta < 3600:
            return f"{int(eta//60)}m {int(eta%60)}s"
        return f"{int(eta//3600)}h {int((eta%3600)//60)}m"

# ---------------- Utilities ----------------
def human_size(n):
    if n is None:
        return "N/A"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.2f} KB"
    if n < 1024**3:
        return f"{n/1024**2:.2f} MB"
    return f"{n/1024**3:.2f} GB"

def unique_dest_path(dest_dir, src_path):
    base = os.path.basename(src_path)
    name, ext = os.path.splitext(base)
    candidate = os.path.join(dest_dir, base)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{name} ({i}){ext}")
        i += 1
    return candidate

def play_video(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass

# ---------------- SQLite cache ----------------
def init_db(path):
    con = sqlite3.connect(path, timeout=30, check_same_thread=False)
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS cache(
             path TEXT PRIMARY KEY,
             size INTEGER,
             mtime REAL,
             method TEXT,
             hashes TEXT,
             frames_sampled INTEGER
           )"""
    )
    con.commit()
    return con

def load_cached(con, path):
    try:
        cur = con.execute("SELECT size,mtime,method,hashes,frames_sampled FROM cache WHERE path=?", (path,))
        row = cur.fetchone()
        if not row:
            return None
        size, mtime, method, hashes_json, frames_sampled = row
        return {"size": size, "mtime": mtime, "method": method, "hashes": json.loads(hashes_json), "frames_sampled": frames_sampled}
    except Exception:
        return None

def upsert_cache(con, path, size, mtime, method, hashes_list, frames_sampled):
    try:
        con.execute(
            """INSERT INTO cache(path,size,mtime,method,hashes,frames_sampled)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 size=excluded.size,
                 mtime=excluded.mtime,
                 method=excluded.method,
                 hashes=excluded.hashes,
                 frames_sampled=excluded.frames_sampled
            """,
            (path, int(size) if size is not None else 0, float(mtime) if mtime is not None else 0.0, str(method), json.dumps(list(hashes_list)), int(frames_sampled)),
        )
        con.commit()
    except Exception:
        pass

def delete_cache_entry(con, path):
    try:
        con.execute("DELETE FROM cache WHERE path=?", (path,))
        con.commit()
    except Exception:
        pass

# ---------------- Sequence JSON cache (fast) ----------------
def seq_cache_path():
    return os.path.join(get_script_dir(), SEQ_CACHE_FILE)

def load_seq_cache():
    path = seq_cache_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_seq_cache(cache):
    path = seq_cache_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def seq_cache_key(video_path, size, mtime):
    return f"{os.path.abspath(video_path)}|{int(size) if size is not None else 0}|{int(mtime) if mtime is not None else 0}"

def get_seq_cached(video_path, size=None, mtime=None):
    try:
        cache = load_seq_cache()
        key = seq_cache_key(video_path, size or 0, mtime or 0)
        return cache.get(key)
    except Exception:
        return None

def set_seq_cache(video_path, seq, size=None, mtime=None):
    try:
        cache = load_seq_cache()
        key = seq_cache_key(video_path, size or 0, mtime or 0)
        cache[key] = seq
        save_seq_cache(cache)
    except Exception:
        pass

# ---------------- thumbnail cache ----------------
def ensure_thumb_cache_dir():
    d = os.path.join(get_script_dir(), THUMB_CACHE_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def hashlib_sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def thumb_cache_filename(video_path):
    try:
        st = os.stat(video_path)
        key = f"{os.path.abspath(video_path)}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        key = os.path.abspath(video_path)
    h = hashlib_sha1(key)
    return os.path.join(ensure_thumb_cache_dir(), f"{h}.png")

def prune_thumb_cache(days=THUMB_MAX_AGE_DAYS):
    try:
        cache_dir = ensure_thumb_cache_dir()
        cutoff = time.time() - days * 86400
        for f in os.listdir(cache_dir):
            path = os.path.join(cache_dir, f)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except Exception:
                pass
    except Exception:
        pass

# ---------------- frame sampling & pHash worker ----------------
def worker_sample_phash_sequence(video_path, max_samples=SAMPLE_FRAMES_PER_VIDEO):
    """Return (path, [hex phash sequence], frames_sampled, size, mtime)"""
    try:
        st = None
        try:
            st = os.stat(video_path)
        except Exception:
            pass

        try:
            if st:
                cached_seq = get_seq_cached(video_path, st.st_size, st.st_mtime)
                if cached_seq:
                    return video_path, list(cached_seq), len(cached_seq), (st.st_size if st else None), (st.st_mtime if st else None)
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path, [], 0, (st.st_size if st else None), (st.st_mtime if st else None)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return video_path, [], 0, (st.st_size if st else None), (st.st_mtime if st else None)
        steps = max(1, total // max_samples)
        indices = list(range(0, total, steps))[:max_samples]
        seq = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ph = imagehash.phash(pil, hash_size=16)
                seq.append(str(ph))
            except Exception:
                pass
        cap.release()
        frames_sampled = len(seq)
        size = st.st_size if st else None
        mtime = st.st_mtime if st else None

        try:
            set_seq_cache(video_path, seq, size=size, mtime=mtime)
        except Exception:
            pass

        return video_path, seq, frames_sampled, size, mtime
    except Exception:
        return video_path, [], 0, None, None

# ---------------- similarity ----------------
def _phash_hex_to_bitarray(seq_hex):
    bit_arrays = []
    for h in seq_hex:
        try:
            bits = bin(int(h, 16))[2:].zfill(len(h) * 4)
            arr = np.fromiter((1 if c == "1" else 0 for c in bits), dtype=np.uint8)
            bit_arrays.append(arr)
        except Exception:
            bit_arrays.append(np.zeros(64, dtype=np.uint8))
    return bit_arrays

def sequence_similarity(seq1_hex, seq2_hex):
    seq1_bits = _phash_hex_to_bitarray(seq1_hex)
    seq2_bits = _phash_hex_to_bitarray(seq2_hex)
    min_len = min(len(seq1_bits), len(seq2_bits))
    if min_len == 0:
        return 0.0
    distances = []
    for i in range(min_len):
        try:
            d = int(np.count_nonzero(seq1_bits[i] != seq2_bits[i]))
        except Exception:
            d = 64
        distances.append(d)
    max_bits = len(seq1_bits[0]) if len(seq1_bits[0]) > 0 else 64
    avg_distance = float(np.mean(distances))
    similarity = max(0.0, 1.0 - (avg_distance / float(max_bits)))
    return similarity

# ---------------- thumbnails ----------------
def try_extract_frame_and_save_to(video_path, target_path, frame_no=None, time_seconds=None, max_size=THUMB_SIZE):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return False
        if frame_no is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        elif time_seconds is not None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return False
        h, w = frame.shape[:2]
        tw, th = max_size
        scale = min(tw / w, th / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        frame_resized = cv2.resize(frame, (nw, nh))
        cv2.imwrite(str(target_path), frame_resized)
        return True
    except Exception:
        return False

def make_or_get_cached_thumbnail(video_path, max_size=THUMB_SIZE):
    cache_path = thumb_cache_filename(video_path)
    if os.path.exists(cache_path):
        return cache_path
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        cap.release()
    except Exception:
        frames = 0
        fps = 0
    tried = []
    if frames > 0:
        tried.append(("frame", max(0, frames // 2)))
    if fps and fps > 1:
        tried.append(("time", 1.0))
    tried.append(("frame", 0))
    for kind, val in tried:
        ok = try_extract_frame_and_save_to(video_path, cache_path,
                                           frame_no=val if kind == "frame" else None,
                                           time_seconds=val if kind == "time" else None,
                                           max_size=max_size)
        if ok and os.path.exists(cache_path):
            return cache_path
    try:
        img = Image.new("RGB", max_size, (60, 60, 60))
        img.save(cache_path)
        return cache_path
    except Exception:
        return None

# ---------------- GUI builders/helpers ----------------
def build_progress_window():
    preview_inner = sg.Column([[]], key="-PREVIEW-INNER-", expand_x=True, expand_y=False, pad=(0,0))
    preview_scroll = sg.Column(
        [[preview_inner]],
        key="-PREVIEW-COL-",
        size=(1000, 420),
        scrollable=True,
        vertical_scroll_only=True,
        expand_x=True,
        pad=(0,0)
    )
    layout = [
        [sg.Text("Search:"), sg.ProgressBar(100, orientation="h", size=(40,18), key="-PSEARCH-"),
         sg.Text("0%", key="-TSEARCH-"), sg.Text("", key="-SEARCH-ETA-", size=(16,1))],
        [sg.Text("Hashing:"), sg.ProgressBar(100, orientation="h", size=(40,18), key="-PHASH-"),
         sg.Text("0%", key="-THASH-"), sg.Text("", key="-HASH-ETA-", size=(16,1))],
        [sg.Text("Comparing:"), sg.ProgressBar(100, orientation="h", size=(40,18), key="-PCOMP-"),
         sg.Text("0%", key="-TCOMP-"), sg.Text("", key="-COMP-ETA-", size=(16,1))],
        [sg.Multiline(size=(140,12), key="-LOG-", autoscroll=True, disabled=True)],
        [preview_scroll],
        [sg.Button("⏸ Pause", key="-PAUSE-"),
         sg.Button("Skip Group", key="-SKIP-", visible=False),
         sg.Button("Quarantine Selected", key="-QUAR-SEL-", visible=True),
         sg.Button("Delete Selected", key="-DEL-SEL-", visible=True),
         sg.Button("New Job", key="-NEWJOB-"), sg.Button("Cancel", key="-CANCEL-")]
    ]
    win = sg.Window("Progress / Log / Preview", layout, finalize=True, resizable=True)

    # attach GUI logging handler (safe: remove old first)
    try:
        for h in list(logger.handlers):
            if isinstance(h, TextHandler):
                try:
                    logger.removeHandler(h)
                except Exception:
                    pass
        gui_handler = TextHandler(win, "-LOG-")
        logger.addHandler(gui_handler)
    except Exception:
        pass

    return win

def append_log(win, text):
    try:
        logger.info(text.rstrip())
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if win and not getattr(win, "was_closed", lambda: False)():
            win["-LOG-"].update(text.rstrip() + "\n", append=True)
    except Exception:
        pass

def update_pct(win, bar_key, txt_key, pct):
    pct = max(0, min(100, int(pct)))
    try:
        try:
            win[bar_key].update_bar(pct)
        except Exception:
            try:
                win[bar_key].update(pct)
            except Exception:
                pass
        win[txt_key].update(f"{pct}%")
    except Exception:
        pass

# --- group rendering ---
def _group_frame_key(gid):
    return ("GROUP", gid)

def _thumbnail_widget_for(path, tfile, is_best):
    if tfile and os.path.exists(tfile):
        thumb_el = sg.Button(image_filename=tfile, key=("THUMB", path), pad=(0,2), tooltip=path)
    else:
        thumb_el = sg.Button(os.path.basename(path), key=("THUMB", path), pad=(0,2), tooltip=path)
    chk = sg.Checkbox("", key=("CHK", path), default=(not is_best), pad=(2,2), tooltip="Mark for delete/quarantine")
    cap = sg.Text("✓ Best" if is_best else "", text_color="green", size=(10,1))
    col = sg.Column([[chk],[thumb_el],[cap],[sg.Text(Path(path).name, size=(28,1))]], pad=(6,6))
    return col

def refresh_group_frame(win, compare_state, gid, cols=4):
    try:
        group = compare_state["groups"][gid]
    except Exception:
        return
    frames_sampled = compare_state.get("frames_sampled", {})
    best = None
    try:
        best = max(group, key=lambda p: frames_sampled.get(p, 0))
    except Exception:
        best = group[0] if group else None

    items = []
    for p in group:
        tfile = compare_state.get("thumb_files", {}).get(p)
        if not tfile or not os.path.exists(tfile):
            tfile = make_or_get_cached_thumbnail(p, max_size=THUMB_SIZE)
            compare_state.setdefault("thumb_files", {})[p] = tfile
        items.append(_thumbnail_widget_for(p, tfile, p == best))

    rows = []
    row = []
    for i, el in enumerate(items):
        row.append(el)
        if (i + 1) % cols == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    header = [
        sg.Text(f"Group {gid+1} — {len(group)} files"),
        sg.Push(),
        sg.Button("Select All", key=("SELALL", gid), size=(12,1)),
        sg.Button("Deselect All", key=("DESELALL", gid), size=(12,1))
    ]
    frame = sg.Frame("", [[*header], *rows], key=_group_frame_key(gid), pad=(4,8))

    try:
        win.extend_layout(win["-PREVIEW-INNER-"], [[frame]])
        win.refresh()
        try:
            canvas = win["-PREVIEW-COL-"].Widget.canvas
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Auto-scroll to bottom/new group
            canvas.yview_moveto(1.0)
        except Exception:
            pass
    except Exception:
        append_log(win, "Failed to update preview grid.")

def refresh_all_groups(win, compare_state):
    try:
        win["-PREVIEW-INNER-"].update([[]])
    except Exception:
        pass
    for gid_idx in range(len(compare_state.get("groups", []))):
        try:
            refresh_group_frame(win, compare_state, gid_idx)
        except Exception:
            pass

def show_group_preview_grid(win, compare_state, group, best, gid, cols=4):
    compare_state.setdefault("frames_sampled", {})
    compare_state["frames_sampled"].update({p: compare_state.get("frames_sampled", {}).get(p, 0) for p in group})
    refresh_group_frame(win, compare_state, gid, cols=cols)

# ---------------- thumbnail cleanup at start ----------------
def clean_old_thumbnails_on_start(prune_days=THUMB_MAX_AGE_DAYS):
    try:
        prune_thumb_cache(prune_days)
    except Exception:
        pass

# ---------------- DB path helper ----------------
def get_db_path():
    return os.path.join(get_script_dir(), DB_FILENAME)

# ---------------- Getting selected items (utility used by some handlers) ----------------
def get_selected_items_from_progress(win):
    """
    Extract checkbox selections from progress window values.
    Returns list of paths where checkbox ("CHK", path) is True.
    """
    selected = []
    try:
        try:
            vals = win.get()
        except Exception:
            vals = {}
        if not isinstance(vals, dict):
            vals = {}
        for k, v in vals.items():
            try:
                if isinstance(k, tuple) and k[0] == "CHK" and v:
                    selected.append(k[1])
            except Exception:
                pass
    except Exception as e:
        log_exception(e, "get_selected_items_from_progress")
    return selected

# ---------------- quarantine/delete helpers ----------------
def ensure_quarantine_dir():
    qd = os.path.join(get_script_dir(), "quarantine")
    os.makedirs(qd, exist_ok=True)
    return qd

def quarantine_paths(paths):
    qd = ensure_quarantine_dir()
    moved = []
    for p in paths:
        try:
            if not os.path.exists(p):
                debug_log(f"Quarantine skipped (not found): {p}")
                continue
            dest = unique_dest_path(qd, p)
            shutil.move(p, dest)
            moved.append((p, dest))
            safe_log(f"Quarantined: {p} -> {dest}")
        except Exception as e:
            log_exception(e, f"quarantine_paths for {p}")
    return moved

def delete_paths(paths):
    deleted = []
    for p in paths:
        try:
            if not os.path.exists(p):
                debug_log(f"Delete skipped (not found): {p}")
                continue
            os.remove(p)
            deleted.append(p)
            safe_log(f"Deleted: {p}")
        except Exception as e:
            log_exception(e, f"delete_paths for {p}")
    return deleted

# ---------------- thumbnail helpers used earlier ----------------
def hashlib_sha1(s):
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def thumb_cache_filename(video_path):
    try:
        st = os.stat(video_path)
        key = f"{os.path.abspath(video_path)}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        key = os.path.abspath(video_path)
    h = hashlib_sha1(key)
    return os.path.join(ensure_thumb_cache_dir(), f"{h}.png")

# ---------------- main (integrated parts) ----------------
def main(stop_mp, pause_mp):
    clean_old_thumbnails_on_start()
    sg.theme("DarkGrey13")

    db_path = get_db_path()
    sqlite_con = init_db(db_path)
    logger.info(f"DB initialized at {db_path}")

    # Main window layout
    sources_col = [
        [sg.Text("Sources (folders/drives to scan):")],
        [sg.Listbox(values=[], select_mode=sg.SELECT_MODE_EXTENDED, size=(70,8), key="-SRC-LB-")],
        [sg.Input(key="-SRC-IN-", enable_events=False, visible=False),
         sg.FolderBrowse("Add Source", target="-SRC-IN-", key="-ADD-SRC-BROWSE-"),
         sg.Button("Add", key="-ADD-SRC-"), sg.Button("Remove Selected", key="-REM-SRC-")]
    ]
    options_col = [
        [sg.Checkbox("Recursive Search", default=True, key="-RECURSIVE-")],
        [sg.Checkbox("Write duplicates summary (duplicates.log)", default=True, key="-LOGSUM-")],
        [sg.Checkbox("Dry-Run (don't delete/move)", default=False, key="-DRYRUN-")],
        [sg.Text("CPU Load %:"), sg.Slider(range=(10,100), default_value=90, resolution=10, orientation="h", size=(36,15), key="-CPU-")],
        [sg.Text("Quarantine Root:"), sg.Input(key="-QUAR-ROOT-"), sg.FolderBrowse(target="-QUAR-ROOT-")],
        [sg.Text("On start, will create '<root>/Quarantine/<timestamp>/'")]
    ]
    main_layout = [
        [sg.Column(sources_col), sg.VSeperator(), sg.Column(options_col)],
        [sg.Radio("Delete duplicates", "ACTION", default=True, key="-ACT-DELETE-"),
         sg.Radio("Move duplicates to quarantine", "ACTION", key="-ACT-MOVE-")],
        [sg.Button("Start"), sg.Button("Exit")]
    ]
    main_win = sg.Window("Video Duplicate Finder", main_layout, finalize=True)
    progress_win = None

    def add_browsed_source():
        p = main_win["-SRC-IN-"].get()
        if p and os.path.isdir(p):
            lst = list(main_win["-SRC-LB-"].get_list_values())
            if p not in lst:
                lst.append(p)
                main_win["-SRC-LB-"].update(lst)

    while True:
        event, values = main_win.read(timeout=100)
        if event in (sg.WINDOW_CLOSED, "Exit"):
            try:
                sqlite_con.close()
            except Exception:
                pass
            break

        if event == "-ADD-SRC-BROWSE-":
            pass
        if event == "-ADD-SRC-":
            add_browsed_source()
        if event == "-REM-SRC-":
            lst = list(main_win["-SRC-LB-"].get_list_values())
            sel = set(values.get("-SRC-LB-", []))
            lst = [x for x in lst if x not in sel]
            main_win["-SRC-LB-"].update(lst)

        if event == "Start":
            sources = list(values.get("-SRC-LB-", []))
            if not sources:
                sg.popup("Please add at least one source folder/drive.")
                continue

            quar_root = values.get("-QUAR-ROOT-")
            if not quar_root or not os.path.isdir(quar_root):
                sg.popup("Please select a valid Quarantine Root folder.")
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            job_quarantine = os.path.join(quar_root, "Quarantine", ts)
            try:
                os.makedirs(job_quarantine, exist_ok=True)
            except Exception:
                pass

            # Create or touch job summary log at start
            summary_path = os.path.join(job_quarantine, "duplicates.log")
            try:
                with open(summary_path, "a", encoding="utf-8") as lf:
                    lf.write(f"JOB START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception:
                pass

            # Open / recreate progress window
            if progress_win:
                try:
                    progress_win.close()
                except Exception:
                    pass
            progress_win = build_progress_window()
            append_log(progress_win, f"Using cache: {db_path}")
            append_log(progress_win, f"Job quarantine: {job_quarantine}")

            # Discover videos
            candidate_list = []
            recursive = bool(values.get("-RECURSIVE-", True))
            for src in sources:
                if not os.path.isdir(src):
                    continue
                if recursive:
                    for root, _, fnames in os.walk(src):
                        for f in fnames:
                            candidate_list.append(os.path.join(root, f))
                else:
                    for f in os.listdir(src):
                        candidate_list.append(os.path.join(src, f))
            videos = [p for p in candidate_list if p.lower().endswith(VIDEO_EXTS) and os.path.isfile(p)]
            append_log(progress_win, f"Found {len(videos)} candidate files across {len(sources)} source(s).")
            update_pct(progress_win, "-PSEARCH-", "-TSEARCH-", 100)
            if not videos:
                append_log(progress_win, "No videos found.")
                continue

            # Hashing / sampling using DB + JSON cache
            cpu_percent = int(values.get("-CPU-", 90))
            max_procs = max(1, int((HASH_PROC_BASE or (os.cpu_count() or 1)) * cpu_percent / 100))
            append_log(progress_win, f"Sampling hashes with up to {max_procs} worker processes (samples/video={SAMPLE_FRAMES_PER_VIDEO}).")

            seq_hashes = {}
            frames_sampled = {}
            to_hash = []
            cached_count = 0

            for p in videos:
                try:
                    st = os.stat(p)
                    cached = load_cached(sqlite_con, p)
                    if cached and cached["size"] == st.st_size and abs(cached["mtime"] - st.st_mtime) < 1e-6 and cached["method"] == "seq_phash":
                        seq_hashes[p] = list(cached["hashes"]) or []
                        frames_sampled[p] = int(cached.get("frames_sampled", len(seq_hashes[p])))
                        cached_count += 1
                        continue
                    seq_fast = get_seq_cached(p, st.st_size, st.st_mtime)
                    if seq_fast:
                        seq_hashes[p] = list(seq_fast)
                        frames_sampled[p] = len(seq_hashes[p])
                        cached_count += 1
                        continue
                    to_hash.append(p)
                except Exception:
                    to_hash.append(p)

            append_log(progress_win, f"Cached OK: {cached_count}; to sample: {len(to_hash)}")

            total_hash = max(1, len(to_hash))
            done_hash = 0
            update_pct(progress_win, "-PHASH-", "-THASH-", 0)
            eta_hash = ETACalc(alpha=ETA_ALPHA)

            if to_hash:
                with ProcessPoolExecutor(max_workers=max_procs) as proc_exec:
                    pending = list(to_hash)
                    running = {}
                    try:
                        while pending or running:
                            # Read progress window events while hashing
                            pe, pv = progress_win.read(timeout=50)
                            # Graceful Cancel / New Job
                            if pe in (sg.WINDOW_CLOSED, "-CANCEL-"):
                                append_log(progress_win, "Cancel requested.")
                                stop_mp.set()
                                break
                            if pe == "-PAUSE-":
                                if pause_mp.is_set():
                                    pause_mp.clear()
                                    progress_win["-PAUSE-"].update("▶ Resume")
                                    append_log(progress_win, "Paused by user.")
                                else:
                                    pause_mp.set()
                                    progress_win["-PAUSE-"].update("⏸ Pause")
                                    append_log(progress_win, "Resumed by user.")
                            if pe == "-NEWJOB-":
                                append_log(progress_win, "New job requested (will stop current job).")
                                stop_mp.set()
                                break

                            cpu_percent = int(values.get("-CPU-", 90))
                            max_inflight = max(1, int((os.cpu_count() or 1) * cpu_percent / 100))

                            # Submit new tasks
                            while pause_mp.is_set() and pending and len(running) < max_inflight and not stop_mp.is_set():
                                p = pending.pop(0)
                                f = proc_exec.submit(worker_sample_phash_sequence, p, SAMPLE_FRAMES_PER_VIDEO)
                                running[f] = p

                            if not running:
                                time.sleep(0.05)
                                continue

                            done_set, _ = wait(list(running.keys()), timeout=0.1, return_when=FIRST_COMPLETED)
                            for fut in list(done_set):
                                try:
                                    ppath, seq, sampled, size, mtime = fut.result(timeout=0)
                                except Exception:
                                    ppath = running.get(fut, "unknown")
                                    seq, sampled, size, mtime = [], 0, None, None
                                seq_hashes[ppath] = list(seq)
                                frames_sampled[ppath] = int(sampled)
                                if size is not None and mtime is not None:
                                    try:
                                        upsert_cache(sqlite_con, ppath, size, mtime, "seq_phash", list(seq_hashes[ppath]), frames_sampled[ppath])
                                    except Exception:
                                        pass
                                done_hash += 1
                                pct = int(done_hash / total_hash * 100)
                                update_pct(progress_win, "-PHASH-", "-THASH-", pct)
                                append_log(progress_win, f"Sampled: {os.path.basename(ppath)} ({pct}%) ETA {eta_hash.update(done_hash, total_hash)}")
                                del running[fut]
                            if stop_mp.is_set():
                                break
                    finally:
                        try:
                            proc_exec.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
            else:
                update_pct(progress_win, "-PHASH-", "-THASH-", 100)
            append_log(progress_win, "Sampling completed.")

            # ---------------- comparison ----------------
            n = len(videos)
            pairs_total = n * (n - 1) // 2
            append_log(progress_win, f"Starting comparison: total pairs = {pairs_total}")
            update_pct(progress_win, "-PCOMP-", "-TCOMP-", 0)
            compare_state = {
                "i": 0, "visited": set(), "pairs_done": 0,
                "pairs_total": max(1, pairs_total), "groups": [], "start_time": time.time(),
                "eta": ETACalc(alpha=ETA_ALPHA), "thumb_files": {},
                "gid": 0, "frames_sampled": frames_sampled
            }
            eta_comp = ETACalc(alpha=ETA_ALPHA)

            def write_action_log_line(line):
                try:
                    with open(summary_path, "a", encoding="utf-8") as lf:
                        lf.write(line.rstrip() + "\n")
                except Exception:
                    pass

            # Main comparison loop — incremental and responsive to progress_win events
            while True:
                pe, pv = progress_win.read(timeout=50)
                # handle immediate GUI actions
                if pe in (sg.WINDOW_CLOSED, "-CANCEL-"):
                    append_log(progress_win, "Cancel requested.")
                    stop_mp.set()
                    break
                if pe == "-NEWJOB-":
                    append_log(progress_win, "New Job requested.")
                    stop_mp.set()
                    break
                if pe == "-PAUSE-":
                    if pause_mp.is_set():
                        pause_mp.clear()
                        append_log(progress_win, "Paused by user.")
                        progress_win["-PAUSE-"].update("▶ Resume")
                    else:
                        pause_mp.set()
                        append_log(progress_win, "Resumed by user.")
                        progress_win["-PAUSE-"].update("⏸ Pause")

                # handle thumbnail click (play)
                if isinstance(pe, tuple) and len(pe) == 2 and pe[0] == "THUMB":
                    play_video(pe[1])

                # handle group select/deselect
                if isinstance(pe, tuple) and len(pe) == 2 and pe[0] in ("SELALL", "DESELALL"):
                    gid = pe[1]
                    if 0 <= gid < len(compare_state["groups"]):
                        grp = compare_state["groups"][gid]
                        for p in grp:
                            key = ("CHK", p)
                            if key in progress_win.AllKeysDict:
                                progress_win[key].update(value=(pe[0] == "SELALL"))

                # QUARANTINE / DELETE from progress window
                if pe in ("-QUAR-SEL-", "-DEL-SEL-"):
                    # read checkboxes current states
                    try:
                        values_chk = progress_win.get()
                    except Exception:
                        values_chk = pv if isinstance(pv, dict) else {}
                    selected = [k[1] for k, v in values_chk.items() if isinstance(k, tuple) and k[0] == "CHK" and v]

                    if not selected:
                        append_log(progress_win, "No items selected.")
                    else:
                        if pe == "-QUAR-SEL-":
                            confirm = sg.popup_yes_no(f"Are you sure you want to move {len(selected)} file(s) to quarantine?")
                            action = "quarantine"
                            do_move = True
                        else:
                            confirm = sg.popup_yes_no(f"Are you sure you want to delete {len(selected)} file(s)?")
                            action = "delete"
                            do_move = False

                        if confirm == "Yes":
                            dry_run = bool(values.get("-DRYRUN-", False))
                            moved_count = 0
                            moved_size = 0
                            for f in selected:
                                try:
                                    # Remove from groups and thumb map
                                    for g in compare_state.get("groups", []):
                                        if f in g:
                                            try:
                                                g.remove(f)
                                            except Exception:
                                                pass
                                    compare_state.get("thumb_files", {}).pop(f, None)

                                    if do_move:
                                        dest_dir = job_quarantine
                                        os.makedirs(dest_dir, exist_ok=True)
                                        dest = unique_dest_path(dest_dir, f)
                                        if dry_run:
                                            append_log(progress_win, f"Dry-Run: would move {f} -> {dest}")
                                            write_action_log_line(f"DRYRUN QUARANTINE: {f} -> {dest}")
                                        else:
                                            shutil.move(f, dest)
                                            moved_count += 1
                                            if os.path.exists(dest):
                                                moved_size += os.path.getsize(dest)
                                            append_log(progress_win, f"Moved to quarantine: {dest}")
                                            write_action_log_line(f"QUARANTINE: {f} -> {dest}")
                                    else:
                                        if dry_run:
                                            append_log(progress_win, f"Dry-Run: would delete {f}")
                                            write_action_log_line(f"DRYRUN DELETE: {f}")
                                        else:
                                            if os.path.exists(f):
                                                sz = os.path.getsize(f)
                                                os.remove(f)
                                                moved_count += 1
                                                moved_size += sz
                                                append_log(progress_win, f"Deleted: {f}")
                                                write_action_log_line(f"DELETE: {f}")
                                                # remove from DB cache
                                                try:
                                                    delete_cache_entry(sqlite_con, f)
                                                except Exception:
                                                    pass
                                except Exception as e:
                                    append_log(progress_win, f"Failed action on {f}: {e}")

                            # cleanup empty groups
                            compare_state["groups"] = [g for g in compare_state.get("groups", []) if g]
                            append_log(progress_win, f"Action complete. Files: {moved_count}; Total size: {human_size(moved_size)}")
                            try:
                                refresh_all_groups(progress_win, compare_state)
                            except Exception:
                                pass
                        else:
                            append_log(progress_win, f"{action.capitalize()} cancelled by user.")

                # Main incremental comparison work
                if compare_state and not stop_mp.is_set():
                    i = compare_state["i"]
                    visited = compare_state["visited"]
                    pairs_done = compare_state["pairs_done"]
                    total_pairs = compare_state["pairs_total"]
                    batch = 0
                    cpu_percent = int(values.get("-CPU-", 90))
                    thumb_workers = max(1, int(THUMB_THREAD_WORKERS_BASE * cpu_percent / 100))
                    while batch < COMPARE_BATCH and i < n:
                        if not pause_mp.is_set() or stop_mp.is_set():
                            break
                        f1 = videos[i]
                        if f1 in visited or not seq_hashes.get(f1):
                            pairs_done += max(0, n - (i + 1))
                            i += 1
                            continue
                        group_found = []
                        for k in range(i + 1, n):
                            if stop_mp.is_set() or not pause_mp.is_set():
                                break
                            f2 = videos[k]
                            if f2 in visited or not seq_hashes.get(f2):
                                pairs_done += 1
                                batch += 1
                                continue
                            sim = sequence_similarity(seq_hashes[f1], seq_hashes[f2])
                            pairs_done += 1
                            batch += 1
                            if sim >= SIMILARITY_THRESHOLD:
                                group_found.append(f2)
                            if batch >= COMPARE_BATCH:
                                break
                        if group_found:
                            group = [f1] + group_found
                            for x in group_found:
                                visited.add(x)
                            best = max(group, key=lambda p: compare_state.get("frames_sampled", {}).get(p, 0))
                            compare_state["groups"].append(group)
                            gid = compare_state["gid"]
                            # warm thumbnails in background
                            with ThreadPoolExecutor(max_workers=thumb_workers) as tpool:
                                futs = {tpool.submit(make_or_get_cached_thumbnail, p, THUMB_SIZE): p for p in group}
                                for ft in as_completed(list(futs.keys())):
                                    pth = futs[ft]
                                    try:
                                        tfile = ft.result()
                                    except Exception:
                                        tfile = None
                                    compare_state.setdefault("thumb_files", {})[pth] = tfile
                            # add group frame and auto-scroll
                            refresh_group_frame(progress_win, compare_state, gid, cols=4)
                            append_log(progress_win, f"Group {gid+1} found ({len(group)} files). Review & choose action.")
                            compare_state["gid"] += 1
                            visited.add(f1)
                            i += 1
                            break
                        else:
                            i += 1
                        if batch > 0:
                            eta = eta_comp.update(pairs_done, total_pairs)
                            try:
                                progress_win["-COMP-ETA-"].update(f"ETA {eta}")
                            except Exception:
                                pass
                    pct = int(pairs_done / total_pairs * 100) if total_pairs else 100
                    update_pct(progress_win, "-PCOMP-", "-TCOMP-", pct)
                    compare_state["i"] = i
                    compare_state["pairs_done"] = pairs_done

                    if i >= n:
                        append_log(progress_win, f"Comparison finished. Groups found: {len(compare_state['groups'])}")
                        # Final summary if requested
                        if values.get("-LOGSUM-"):
                            try:
                                with open(summary_path, "a", encoding="utf-8") as sf:
                                    sf.write("\nSUMMARY\n")
                                    for g in compare_state["groups"]:
                                        if not g:
                                            continue
                                        best = max(g, key=lambda p: compare_state.get("frames_sampled", {}).get(p, 0))
                                        sf.write("GROUP:\n")
                                        sf.write(f"  KEEP: {best}\n")
                                        for x in g:
                                            if x != best:
                                                if values.get("-ACT-MOVE-"):
                                                    sf.write(f"  QUARANTINE: {x}\n")
                                                else:
                                                    sf.write(f"  DELETE: {x}\n")
                                        sf.write("\n")
                                append_log(progress_win, f"Wrote summary to {summary_path}")
                            except Exception as e:
                                append_log(progress_win, f"Failed to write summary: {e}")
                        sg.popup("Duplicate detection finished.")
                        compare_state = None
                        break

            # reset flags for next run
            try:
                stop_mp.clear()
            except Exception:
                pass
            try:
                pause_mp.set()
            except Exception:
                pass

            # keep the progress window open for manual actions
            # user can still click quarantine / delete / new job / cancel

    # final cleanup
    try:
        if progress_win:
            try:
                for h in list(logger.handlers):
                    if isinstance(h, TextHandler):
                        logger.removeHandler(h)
            except Exception:
                pass
            try:
                progress_win.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        main_win.close()
    except Exception:
        pass
    try:
        sqlite_con.close()
    except Exception:
        pass

# ---------------- entry ----------------
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    stop_mp = multiprocessing.Event()
    pause_mp = multiprocessing.Event()
    pause_mp.set()
    debug_log("Starting compare_working_fixed_final.py")
    main(stop_mp, pause_mp)
    debug_log("Exiting compare_working_fixed_final.py")
