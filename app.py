"""
NeuroWatch — Human–AI Hybrid Mental Health Monitoring System
With full Login / Sign-Up authentication backed by SQLite
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import time
import requests
import sqlite3
import hashlib
import hmac
import secrets
import re
from datetime import datetime
from typing import Tuple
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Kaggle auto-download for Streamlit Cloud deployment ───────────────────────
import zipfile
import urllib.request
import json
import base64

WESAD_PATH = "/tmp/wesad/WESAD"

def download_wesad():
    """
    Download and extract the WESAD dataset from Kaggle using their
    REST API directly — no subprocess pip install required.
    Works on Streamlit Cloud where runtime pip is blocked.
    """
    if os.path.exists(WESAD_PATH):
        return True, "Dataset already downloaded."

    # Pull credentials from Streamlit secrets → env vars
    kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
    kaggle_key  = os.environ.get("KAGGLE_KEY", "")

    if not kaggle_user or not kaggle_key:
        return False, (
            "Kaggle credentials not found. "
            "Add KAGGLE_USERNAME and KAGGLE_KEY to Streamlit Cloud secrets."
        )

    try:
        os.makedirs("/tmp/wesad", exist_ok=True)

        zip_path = "/tmp/wesad/wesad.zip"

        # ── Kaggle dataset download API (no CLI needed) ───────────────────────
        dataset_owner = "orvile"
        dataset_name  = "wesad-wearable-stress-affect-detection-dataset"
        api_url = (
            f"https://www.kaggle.com/api/v1/datasets/download/"
            f"{dataset_owner}/{dataset_name}"
        )

        # Basic auth: base64(username:key)
        credentials = base64.b64encode(
            f"{kaggle_user}:{kaggle_key}".encode()
        ).decode()

        req = urllib.request.Request(
            api_url,
            headers={
                "Authorization": f"Basic {credentials}",
                "User-Agent":    "python-urllib",
            }
        )

        # Stream download to disk (dataset is ~1 GB — stream in chunks)
        with urllib.request.urlopen(req, timeout=300) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(zip_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            return False, "Download produced an empty or invalid file. Check your Kaggle credentials."

        # ── Extract zip ───────────────────────────────────────────────────────
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/tmp/wesad")
        os.remove(zip_path)

        # ── Verify extraction ─────────────────────────────────────────────────
        if os.path.exists(WESAD_PATH):
            return True, "✅ WESAD dataset downloaded and extracted successfully."

        # Some zips unpack with a nested folder — find it
        for root, dirs, files in os.walk("/tmp/wesad"):
            for d in dirs:
                candidate = os.path.join(root, d, "WESAD")
                if os.path.exists(candidate):
                    import shutil
                    shutil.move(candidate, WESAD_PATH)
                    return True, "✅ WESAD dataset ready."

        return False, (
            "Extraction completed but WESAD folder not found inside the zip. "
            "The dataset structure may have changed on Kaggle."
        )

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Kaggle credentials are invalid. Check KAGGLE_USERNAME and KAGGLE_KEY in secrets."
        if e.code == 403:
            return False, "Access denied. Make sure you have accepted the dataset terms on Kaggle.com."
        return False, f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"Network error: {str(e.reason)}"
    except zipfile.BadZipFile:
        return False, "Downloaded file is not a valid zip. The Kaggle API may have returned an error page."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# ── Load Streamlit secrets into environment variables ─────────────────────────
# This makes KAGGLE_USERNAME and KAGGLE_KEY available via os.environ
# both locally (if set in .streamlit/secrets.toml) and on Streamlit Cloud
try:
    if "KAGGLE_USERNAME" in st.secrets:
        os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    if "KAGGLE_KEY" in st.secrets:
        os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
except Exception:
    pass  # secrets not configured — local mode, user sets path manually

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroWatch — AI Patient Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ═════════════════════════════════════════════════════════════════════════════

DB_PATH = "neurowatch_users.db"

def get_db():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create users and session_log tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            username      TEXT    NOT NULL UNIQUE,
            role          TEXT    NOT NULL DEFAULT 'clinician',
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            last_login    TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            username    TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            detail      TEXT,
            timestamp   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 password hash."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex()

def create_user(full_name, email, username, role, password) -> Tuple[bool, str]:
    """Insert a new user. Returns (success, message)."""
    salt = secrets.token_hex(32)
    pw_hash = hash_password(password, salt)
    try:
        conn = get_db()
        conn.execute(
            """INSERT INTO users
               (full_name, email, username, role, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (full_name, email.lower().strip(), username.strip(), role,
             pw_hash, salt, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        if "email" in str(e):
            return False, "An account with that email already exists."
        if "username" in str(e):
            return False, "That username is already taken."
        return False, "Registration failed. Please try again."

def verify_user(username_or_email: str, password: str):
    """Return user Row if credentials are valid, else None."""
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? OR email=?",
        (username_or_email.strip(), username_or_email.strip().lower()),
    ).fetchone()
    conn.close()
    if user is None:
        return None
    expected = hash_password(password, user["salt"])
    if hmac.compare_digest(expected, user["password_hash"]):
        return user
    return None

def update_last_login(user_id: int):
    conn = get_db()
    conn.execute(
        "UPDATE users SET last_login=? WHERE id=?",
        (datetime.now().isoformat(), user_id),
    )
    conn.commit()
    conn.close()

def log_action(user_id: int, username: str, action: str, detail: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO session_log (user_id, username, action, detail, timestamp) VALUES (?,?,?,?,?)",
        (user_id, username, action, detail, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def get_all_users():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, full_name, email, username, role, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return rows

def get_session_log(limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT username, action, detail, timestamp FROM session_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows

def delete_user(user_id: int):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

# Initialise DB on every cold start
init_db()

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def validate_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", email))

def validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", pw):
        return False, "Password must contain at least one number."
    return True, ""

def validate_username(un: str) -> Tuple[bool, str]:
    if len(un) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_]+$", un):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, ""

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (auth pages + main dashboard)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0B0F1A;
    color: #E2E8F0;
}

/* ── Auth card ── */
.auth-wrap {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 40px 0 60px 0;
}
.auth-card {
    background: #111827;
    border: 1px solid #1E3A5F;
    border-radius: 18px;
    padding: 44px 48px;
    width: 100%;
    max-width: 480px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.6);
}
.auth-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #38BDF8;
    letter-spacing: -1px;
    margin-bottom: 4px;
    text-align: center;
}
.auth-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4A7FA5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 32px;
}
.auth-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 6px;
}
.auth-subtitle {
    font-size: 0.83rem;
    color: #64748B;
    margin-bottom: 28px;
}
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4A7FA5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.divider {
    height: 1px;
    background: #1E2D40;
    margin: 24px 0;
}
.error-box {
    background: #450a0a;
    border: 1px solid #991b1b;
    border-radius: 8px;
    padding: 10px 14px;
    color: #fca5a5;
    font-size: 0.82rem;
    margin-bottom: 16px;
}
.success-box {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 10px 14px;
    color: #86efac;
    font-size: 0.82rem;
    margin-bottom: 16px;
}
.pw-hint {
    font-size: 0.72rem;
    color: #334155;
    margin-top: 4px;
}

/* ── Main dashboard ── */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D40;
}
section[data-testid="stSidebar"] * { color: #CBD5E0 !important; }

.nw-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 8px 0;
    border-bottom: 1px solid #1E3A5F;
    margin-bottom: 24px;
}
.nw-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #38BDF8;
    letter-spacing: -1px;
}
.nw-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4A7FA5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}
.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0D1B2A;
    border: 1px solid #1E3A5F;
    border-radius: 100px;
    padding: 5px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #38BDF8;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
}
.badge-stable   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-stress   { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
.badge-admin    { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; }
.badge-clinician{ background: #0c2340; color: #93c5fd; border: 1px solid #1d4ed8; }
.badge-researcher{background: #134e4a; color: #5eead4; border: 1px solid #0f766e; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 20px 0 12px 0;
    border-left: 3px solid #38BDF8;
    padding-left: 10px;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4A7FA5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.report-box {
    background: #0D1117;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 24px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    color: #94A3B8;
    white-space: pre-wrap;
}
.info-banner {
    background: #0c2340;
    border: 1px solid #1E3A5F;
    border-left: 4px solid #38BDF8;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #93C5FD;
    margin-bottom: 16px;
}
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1E2D40;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #4A7FA5 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #F8FAFC !important; font-family: 'Syne', sans-serif !important; }
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0B0F1A; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

defaults = {
    "authenticated": False,
    "user_id":       None,
    "username":      None,
    "full_name":     None,
    "role":          None,
    "auth_page":     "login",   # "login" | "signup"
    "model":         None,
    "scaler":        None,
    "df_all":        None,
    "report":        None,
    "cm":            None,
    "sim_log":       None,
    "loaded_subjects": [],
    "live_log":      None,
    "live_api_url":  "http://localhost:8000",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ═════════════════════════════════════════════════════════════════════════════

def render_login():
    st.markdown("""
    <div class="auth-logo">🧠 NeuroWatch</div>
    <div class="auth-tagline">Human–AI Hybrid Mental Health Monitoring</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign in to access the monitoring dashboard</div>', unsafe_allow_html=True)

        if "login_error" in st.session_state and st.session_state.login_error:
            st.markdown(f'<div class="error-box">⚠ {st.session_state.login_error}</div>',
                        unsafe_allow_html=True)
            st.session_state.login_error = ""

        st.markdown('<div class="field-label">Username or Email</div>', unsafe_allow_html=True)
        identifier = st.text_input("Username or Email", label_visibility="collapsed",
                                   placeholder="Enter your username or email",
                                   key="login_identifier")

        st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
        password = st.text_input("Password", label_visibility="collapsed",
                                 type="password", placeholder="Enter your password",
                                 key="login_password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True, type="primary"):
            if not identifier or not password:
                st.session_state.login_error = "Please fill in all fields."
                st.rerun()
            else:
                user = verify_user(identifier, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id   = user["id"]
                    st.session_state.username  = user["username"]
                    st.session_state.full_name = user["full_name"]
                    st.session_state.role      = user["role"]
                    update_last_login(user["id"])
                    log_action(user["id"], user["username"], "login")
                    st.rerun()
                else:
                    st.session_state.login_error = "Invalid username/email or password."
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.83rem;color:#64748B">Don\'t have an account?</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account", use_container_width=True):
            st.session_state.auth_page = "signup"
            st.rerun()


def render_signup():
    st.markdown("""
    <div class="auth-logo">🧠 NeuroWatch</div>
    <div class="auth-tagline">Human–AI Hybrid Mental Health Monitoring</div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="auth-title">Create your account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Join NeuroWatch to access the monitoring dashboard</div>',
                    unsafe_allow_html=True)

        if "signup_error" in st.session_state and st.session_state.signup_error:
            st.markdown(f'<div class="error-box">⚠ {st.session_state.signup_error}</div>',
                        unsafe_allow_html=True)
            st.session_state.signup_error = ""

        if "signup_success" in st.session_state and st.session_state.signup_success:
            st.markdown('<div class="success-box">✓ Account created! You can now sign in.</div>',
                        unsafe_allow_html=True)
            st.session_state.signup_success = False

        st.markdown('<div class="field-label">Full Name</div>', unsafe_allow_html=True)
        full_name = st.text_input("Full Name", label_visibility="collapsed",
                                  placeholder="Dr. Jane Smith", key="reg_name")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="field-label">Username</div>', unsafe_allow_html=True)
            username = st.text_input("Username", label_visibility="collapsed",
                                     placeholder="dr_smith", key="reg_username")
        with c2:
            st.markdown('<div class="field-label">Role</div>', unsafe_allow_html=True)
            role = st.selectbox("Role", ["clinician", "researcher", "admin"],
                                label_visibility="collapsed", key="reg_role")

        st.markdown('<div class="field-label">Email Address</div>', unsafe_allow_html=True)
        email = st.text_input("Email", label_visibility="collapsed",
                              placeholder="jane@hospital.org", key="reg_email")

        st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
        pw1 = st.text_input("Password", label_visibility="collapsed", type="password",
                             placeholder="Min 8 chars, 1 uppercase, 1 number", key="reg_pw1")
        st.markdown('<div class="pw-hint">Minimum 8 characters · at least one uppercase letter · at least one number</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="field-label">Confirm Password</div>', unsafe_allow_html=True)
        pw2 = st.text_input("Confirm Password", label_visibility="collapsed", type="password",
                             placeholder="Repeat your password", key="reg_pw2")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True, type="primary"):
            # Validate
            err = None
            if not all([full_name, username, email, pw1, pw2]):
                err = "Please fill in all fields."
            elif not validate_email(email):
                err = "Please enter a valid email address."
            else:
                ok, msg = validate_username(username)
                if not ok:
                    err = msg
                else:
                    ok, msg = validate_password(pw1)
                    if not ok:
                        err = msg
                    elif pw1 != pw2:
                        err = "Passwords do not match."

            if err:
                st.session_state.signup_error = err
                st.rerun()
            else:
                success, msg = create_user(full_name, email, username, role, pw1)
                if success:
                    st.session_state.signup_success = True
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.session_state.signup_error = msg
                    st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.83rem;color:#64748B">Already have an account?</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to Sign In", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING — show auth or dashboard
# ═════════════════════════════════════════════════════════════════════════════

if not st.session_state.authenticated:
    if st.session_state.auth_page == "signup":
        if "signup_error"   not in st.session_state: st.session_state.signup_error   = ""
        if "signup_success" not in st.session_state: st.session_state.signup_success = False
        render_signup()
    else:
        if "login_error" not in st.session_state: st.session_state.login_error = ""
        render_login()
    st.stop()   # <── nothing below runs until authenticated


# ═════════════════════════════════════════════════════════════════════════════
# EVERYTHING BELOW IS THE PROTECTED DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
ECG_FS = 700;  EDA_FS = 4;  RESP_FS = 700;  TEMP_FS = 700
WINDOW_SECONDS = 60
FEATURES = [
    "heart_rate", "sdnn", "rmssd",
    "eda_mean", "eda_std", "eda_peaks", "arousal_index",
    "resp_mean", "resp_std", "temp_mean"
]

# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass_filter(sig, low, high, fs):
    b, a = butter(2, [low / (fs * 0.5), high / (fs * 0.5)], btype='band')
    return filtfilt(b, a, sig)

def extract_ecg_features(ecg, fs=ECG_FS):
    peaks, _ = find_peaks(ecg, distance=fs*0.5)
    if len(peaks) < 2:
        return {"heart_rate": 70, "sdnn": 0.05, "rmssd": 0.04}
    rr = np.diff(peaks) / fs
    return {
        "heart_rate": 60/np.mean(rr),
        "sdnn":       np.std(rr),
        "rmssd":      np.sqrt(np.mean(np.square(np.diff(rr))))
    }

def extract_eda_features(eda):
    eda = np.array(eda)
    m, s = np.mean(eda), np.std(eda)
    peaks, _ = find_peaks(eda, height=m+0.1*s, distance=4)
    return {"eda_mean": m, "eda_std": s,
            "eda_peaks": len(peaks), "arousal_index": len(peaks)/len(eda)}

def extract_resp_features(resp):
    return {"resp_mean": np.mean(resp), "resp_std": np.std(resp)}

def extract_temp_features(temp):
    return {"temp_mean": np.mean(temp)}

# ── WESAD pipeline ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(base_path):
    subjects = [s for s in os.listdir(base_path) if s.startswith("S")]
    dataset, loaded = [], []
    for subject in subjects:
        fp = f"{base_path}/{subject}/{subject}.pkl"
        try:
            with open(fp, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception:
            continue
        chest  = data["signal"]["chest"]
        ecg    = chest["ECG"].flatten();   eda  = chest["EDA"].flatten()
        resp   = chest["Resp"].flatten();  temp = chest["Temp"].flatten()
        labels = data["label"].flatten()
        ecg_w  = ECG_FS*WINDOW_SECONDS;   eda_w  = EDA_FS*WINDOW_SECONDS
        resp_w = RESP_FS*WINDOW_SECONDS;  temp_w = TEMP_FS*WINDOW_SECONDS
        windows = min(len(ecg)//ecg_w, len(eda)//eda_w,
                      len(resp)//resp_w, len(temp)//temp_w)
        for i in range(windows):
            feats = {
                **extract_ecg_features(ecg[i*ecg_w:(i+1)*ecg_w]),
                **extract_eda_features(eda[i*eda_w:(i+1)*eda_w]),
                **extract_resp_features(resp[i*resp_w:(i+1)*resp_w]),
                **extract_temp_features(temp[i*temp_w:(i+1)*temp_w]),
            }
            lbl = int(np.round(np.mean(labels[i*ecg_w:(i+1)*ecg_w])))
            dataset.append({**feats, "label": 1 if lbl == 2 else 0})
        loaded.append(subject)

    df = pd.DataFrame(dataset)
    X, y = df[FEATURES], df["label"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2,
                                               stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=600, max_depth=14,
                                   min_samples_split=4, class_weight="balanced",
                                   random_state=42)
    model.fit(X_tr, y_tr)
    rpt = classification_report(y_te, model.predict(X_te), output_dict=True)
    cm  = confusion_matrix(y_te, model.predict(X_te))
    return model, scaler, loaded, df, rpt, cm

@st.cache_data(show_spinner=False)
def load_subject_signals(base_path, subject):
    fp = f"{base_path}/{subject}/{subject}.pkl"
    with open(fp, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    chest = data["signal"]["chest"]
    return (chest["ECG"].flatten(), chest["EDA"].flatten(),
            chest["Resp"].flatten(), chest["Temp"].flatten(),
            data["label"].flatten())

def extract_realtime_features(ecg, eda, resp, temp, max_windows=60):
    ecg_w  = ECG_FS*WINDOW_SECONDS;  eda_w  = EDA_FS*WINDOW_SECONDS
    resp_w = RESP_FS*WINDOW_SECONDS; temp_w = TEMP_FS*WINDOW_SECONDS
    n = min(len(ecg)//ecg_w, len(eda)//eda_w,
            len(resp)//resp_w, len(temp)//temp_w, max_windows)
    out = []
    for i in range(n):
        out.append({
            **extract_ecg_features(ecg[i*ecg_w:(i+1)*ecg_w]),
            **extract_eda_features(eda[i*eda_w:(i+1)*eda_w]),
            **extract_resp_features(resp[i*resp_w:(i+1)*resp_w]),
            **extract_temp_features(temp[i*temp_w:(i+1)*temp_w]),
        })
    return out

def predict_minute(model, scaler, features):
    x = pd.DataFrame([features])[FEATURES]
    xs = scaler.transform(x)
    proba = model.predict_proba(xs)[0]
    pred  = model.predict(xs)[0]
    return ("Distress Detected" if pred == 1 else "Stable"), proba[pred], proba[1]

def generate_report(features, state, confidence, minute):
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    return f"""
╔══════════════════════════════════════════════════╗
   AI HOURLY CLINICAL REPORT
   Human–AI Hybrid Monitoring System — NeuroWatch
   Clinician: {st.session_state.full_name}
╚══════════════════════════════════════════════════╝

  Generated     : {ts}
  Monitoring    : {minute} minutes of continuous data

──────────────────────────────────────────────────
  AGGREGATED PHYSIOLOGICAL ANALYSIS
──────────────────────────────────────────────────

  Heart Rate             : {features['heart_rate']:.2f} bpm
  HRV — SDNN             : {features['sdnn']:.4f} s
  HRV — RMSSD            : {features['rmssd']:.4f} s
  Mean EDA               : {features['eda_mean']:.3f} µS
  EDA Variability        : {features['eda_std']:.3f}
  EDA Peaks              : {features['eda_peaks']:.1f}
  Arousal Index          : {features['arousal_index']:.4f}
  Respiration Mean       : {features['resp_mean']:.3f}
  Respiration Variability: {features['resp_std']:.3f}
  Skin Temperature       : {features['temp_mean']:.2f} °C

──────────────────────────────────────────────────
  AI FINAL ASSESSMENT
──────────────────────────────────────────────────

  Predicted Emotional State : {state}
  Model Confidence          : {confidence:.2%}

──────────────────────────────────────────────────
  CLINICAL INTERPRETATION
──────────────────────────────────────────────────

  AI has analyzed continuous physiological signals.
  Autonomic nervous system patterns, cardiovascular
  variability, electrodermal response, respiration,
  and temperature regulation were assessed jointly.

  This report is a decision-support tool only.
  Final clinical judgement must always be made
  by a qualified healthcare professional.

══════════════════════════════════════════════════
  Human–AI Hybrid System · AI assists, humans decide
══════════════════════════════════════════════════
"""

# ── Chart helpers ─────────────────────────────────────────────────────────────
DARK_BG = "#0B0F1A"; CARD_BG = "#111827"; GRID_COL = "#1E2D40"
TEXT_COL = "#64748B"; ACCENT = "#38BDF8"; DANGER = "#F87171"
WARNING = "#FBBF24"; SUCCESS = "#4ADE80"

def _dark(ax, fig):
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color("#CBD5E0")
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)

def plot_risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.3,1.3); ax.axis("off")
    theta_bg = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color=GRID_COL, linewidth=10, solid_capstyle="round")
    theta_f = np.linspace(0, np.pi*prob, 200)
    colour = SUCCESS if prob<0.4 else (WARNING if prob<0.7 else DANGER)
    ax.plot(np.cos(theta_f), np.sin(theta_f), color=colour, linewidth=10, solid_capstyle="round")
    angle = np.pi*(1-prob)
    ax.annotate("", xy=(0.72*np.cos(angle), 0.72*np.sin(angle)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="#F8FAFC", lw=2))
    label = "Stable" if prob<0.4 else ("Moderate" if prob<0.7 else "High Risk")
    ax.text(0,-0.2, f"{label}  ·  {prob:.0%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=colour, fontfamily="monospace")
    ax.text(-1.1,0,"0%",ha="center",fontsize=7,color=TEXT_COL)
    ax.text(1.1,0,"100%",ha="center",fontsize=7,color=TEXT_COL)
    ax.text(0,1.15,"STRESS RISK",ha="center",fontsize=7,color=TEXT_COL,
            fontfamily="monospace",fontstyle="italic")
    return fig

def plot_timeseries(df, col, title, colour, ylabel):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    _dark(ax, fig)
    ax.plot(df["minute"], df[col], color=colour, linewidth=1.8, alpha=0.9)
    ax.fill_between(df["minute"], df[col], alpha=0.12, color=colour)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Minute", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    fig.tight_layout(); return fig

def plot_feature_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _dark(ax, fig)
    imp_median = float(imp.median())
    imp_vals = imp.values.tolist()
    colours = [ACCENT if v > imp_median else GRID_COL for v in imp_vals]
    ax.barh(imp.index, imp_vals, color=colours, edgecolor="none", height=0.6)
    ax.set_title("Feature Importance", fontsize=10, pad=8)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=8)
    fig.tight_layout()
    return fig

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(3.5, 3)); _dark(ax, fig)
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Stress","Stress"],fontsize=8)
    ax.set_yticklabels(["Non-Stress","Stress"],fontsize=8)
    ax.set_xlabel("Predicted",fontsize=8); ax.set_ylabel("Actual",fontsize=8)
    ax.set_title("Confusion Matrix",fontsize=10,pad=8)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    fontsize=14,color="white",fontweight="bold")
    fig.tight_layout(); return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # User info
    role_badge = {
        "admin": "badge-admin",
        "clinician": "badge-clinician",
        "researcher": "badge-researcher",
    }.get(st.session_state.role, "badge-clinician")

    st.markdown(f"""
    <div style="padding:12px 0 16px 0;border-bottom:1px solid #1E2D40;margin-bottom:16px">
      <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#4A7FA5;
                  letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">
        Signed in as
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                  color:#F1F5F9;margin-bottom:4px">
        {st.session_state.full_name}
      </div>
      <span class="badge {role_badge}">@{st.session_state.username}</span>
      <span class="badge {role_badge}" style="margin-left:6px">{st.session_state.role}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")

    # Auto-detect deployment vs local
    is_cloud = os.environ.get("KAGGLE_USERNAME", "") != ""

    if is_cloud:
        # On Streamlit Cloud — use fixed download path
        base_path = WESAD_PATH
        st.markdown(
            "<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
            "color:#4A7FA5;margin-bottom:8px'>WESAD Path (auto)</div>",
            unsafe_allow_html=True
        )
        st.code(base_path, language=None)

        if not os.path.exists(WESAD_PATH):
            if st.button("⬇️ Download WESAD Dataset", use_container_width=True):
                with st.spinner("Downloading WESAD from Kaggle… this takes 1–2 minutes."):
                    ok, msg = download_wesad()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                    st.stop()
        else:
            st.success("✅ WESAD dataset ready")
    else:
        # Local — let user type their own path
        base_path = st.text_input("WESAD Base Path", value="/content/wesad/WESAD")

    load_btn = st.button("🚀 Load Dataset & Train Model", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧪 Simulation")
    sim_subject = st.selectbox("Simulate Patient (Subject)",
                                ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11"])
    sim_minutes = st.slider("Minutes to Simulate", 5, 60, 20)
    run_sim     = st.button("▶ Run Live Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔴 Live Watch (Phase 2)")
    live_api_url = st.text_input("Live API URL", value=st.session_state.live_api_url,
                                  key="live_api_url_input",
                                  help="Where api.py is running — start it separately "
                                       "with `uvicorn api:app --port 8000`.")
    st.session_state.live_api_url = live_api_url

    poll_seconds = st.slider("Poll Interval (sec)", 1, 15, 2, key="poll_seconds")
    poll_cycles  = st.slider("Cycles to Poll", 5, 120, 30, key="poll_cycles")
    start_live   = st.button("▶ Start Live Feed", use_container_width=True)

    cal1, cal2 = st.columns(2)
    with cal1:
        push_cal = st.button("📡 Push Calibration", use_container_width=True,
                              disabled=st.session_state.model is None,
                              help="Sends WESAD mean/std per feature to the API so "
                                   "watch_simulation.py drifts around realistic values.")
    with cal2:
        reset_live = st.button("🗑️ Reset Live Buffer", use_container_width=True)

    if push_cal:
        try:
            stats = st.session_state.df_all[FEATURES].agg(["mean", "std"]).to_dict()
            stats = {f: {"mean": float(v["mean"]), "std": float(v["std"])} for f, v in stats.items()}
            r = requests.post(f"{live_api_url}/calibration", json={"stats": stats}, timeout=4)
            if r.status_code == 200:
                st.success("✅ Calibration pushed to live API.")
            else:
                st.error(f"API rejected calibration: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API at {live_api_url}: {e}")

    if reset_live:
        try:
            r = requests.delete(f"{live_api_url}/reset", timeout=4)
            if r.status_code == 200:
                st.session_state.live_log = None
                st.success("✅ Live buffer cleared.")
            else:
                st.error(f"API error: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API at {live_api_url}: {e}")

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        log_action(st.session_state.user_id, st.session_state.username, "logout")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown(
        "<div style='font-size:0.7rem;color:#334155;line-height:1.6;margin-top:12px'>"
        "NeuroWatch v1.0<br>Human–AI Hybrid System<br>AI supports · Clinician decides"
        "</div>", unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="nw-header">
  <div style="flex:1">
    <div class="nw-logo">🧠 NeuroWatch</div>
    <div class="nw-subtitle">Human–AI Hybrid Mental Health Monitoring · WESAD · Bedridden Patient Support</div>
  </div>
  <div>
    <div class="user-pill">👤 {st.session_state.full_name} &nbsp;·&nbsp; {st.session_state.role}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
  ⚠️ <strong>Clinical Disclaimer:</strong> This system provides AI-assisted decision support only.
  All clinical decisions must be made by a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN
# ═════════════════════════════════════════════════════════════════════════════

if load_btn:
    # If running on cloud and dataset not yet downloaded, download first
    if not os.path.exists(base_path):
        if os.environ.get("KAGGLE_USERNAME", ""):
            with st.spinner("Downloading WESAD dataset from Kaggle first…"):
                ok, msg = download_wesad()
            if not ok:
                st.error(msg)
                st.stop()
            else:
                st.success(msg)
        else:
            st.error(f"Path not found: `{base_path}` — Check your WESAD Base Path in the sidebar.")
            st.stop()

    with st.spinner("Loading WESAD dataset and training model…"):
        model, scaler, subjects, df_all, rpt, cm = load_and_train(base_path)
        st.session_state.model   = model
        st.session_state.scaler  = scaler
        st.session_state.df_all  = df_all
        st.session_state.report  = rpt
        st.session_state.cm      = cm
        st.session_state.loaded_subjects = subjects
    log_action(st.session_state.user_id, st.session_state.username,
               "model_trained", f"{len(subjects)} subjects, {len(df_all)} windows")
    st.success(f"✅ Model trained on {len(subjects)} subjects · {len(df_all)} windows")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tabs = ["📊 Live Monitor", "🔬 Model Evaluation", "📋 Clinical Report",
        "🔴 Live Watch", "ℹ️ About System"]
if st.session_state.role == "admin":
    tabs.append("🛡️ Admin Panel")

tab_objects = st.tabs(tabs)
tab1, tab2, tab3, tab_live, tab4 = tab_objects[:5]
tab_admin = tab_objects[5] if len(tab_objects) > 5 else None

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Monitor
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if st.session_state.model is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#334155">
          <div style="font-size:3rem">🧠</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.3rem;color:#38BDF8;margin:12px 0">
            Load the dataset to begin monitoring
          </div>
          <div style="font-size:0.85rem">Use the sidebar → <b>Load Dataset & Train Model</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if run_sim:
            fp_check = f"{base_path}/{sim_subject}/{sim_subject}.pkl"
            if not os.path.exists(fp_check):
                st.error(f"Subject {sim_subject} not found.")
            else:
                ecg, eda, resp, temp, _ = load_subject_signals(base_path, sim_subject)
                all_feat = extract_realtime_features(ecg, eda, resp, temp,
                                                     max_windows=sim_minutes)
                log, prog_bar = [], st.progress(0, text="Simulating…")
                gauge_ph = st.empty()
                c1, c2   = st.columns(2)
                hr_ph, ep_ph = c1.empty(), c2.empty()

                for i, feat in enumerate(all_feat):
                    state, conf, sp = predict_minute(
                        st.session_state.model, st.session_state.scaler, feat)
                    log.append({"minute": i+1, "heart_rate": feat["heart_rate"],
                                "eda_mean": feat["eda_mean"],
                                "stress_prob": sp, "confidence": conf, "state": state})
                    df_log = pd.DataFrame(log)

                    with gauge_ph.container():
                        gc = st.columns(5)
                        gc[0].metric("Heart Rate", f"{feat['heart_rate']:.1f} bpm")
                        gc[1].metric("EDA",        f"{feat['eda_mean']:.3f} µS")
                        gc[2].metric("Resp",       f"{feat['resp_mean']:.3f}")
                        gc[3].metric("Temp",       f"{feat['temp_mean']:.1f} °C")
                        gc[4].metric("SDNN",       f"{feat['sdnn']:.4f} s")

                        rg1, rg2 = st.columns([1,2])
                        with rg1:
                            st.pyplot(plot_risk_gauge(sp), use_container_width=True)
                        with rg2:
                            bcls = "badge-stress" if state=="Distress Detected" else "badge-stable"
                            st.markdown(f"""
                            <div style="padding:20px 0">
                              <div class="metric-label">AI Assessment — Minute {i+1}</div>
                              <div style="margin:10px 0">
                                <span class="badge {bcls}">{state}</span>
                              </div>
                              <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                          color:#4A7FA5;margin-top:8px">
                                Confidence: <span style="color:#38BDF8">{conf:.1%}</span>
                              </div>
                              <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                                          color:#334155;margin-top:4px">
                                Stress Probability: <span style="color:#FBBF24">{sp:.1%}</span>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    with hr_ph:
                        st.pyplot(plot_timeseries(df_log,"heart_rate",
                                                  "Heart Rate (bpm)",ACCENT,"BPM"),
                                  use_container_width=True)
                    with ep_ph:
                        st.pyplot(plot_timeseries(df_log,"stress_prob",
                                                  "Stress Probability",DANGER,"Probability"),
                                  use_container_width=True)

                    prog_bar.progress((i+1)/len(all_feat),
                                      text=f"Minute {i+1}/{len(all_feat)} — {state}")
                    time.sleep(0.15)

                prog_bar.empty()
                st.session_state.sim_log = pd.DataFrame(log)
                log_action(st.session_state.user_id, st.session_state.username,
                           "simulation_run",
                           f"subject={sim_subject}, minutes={len(all_feat)}")
                st.success(f"✅ Simulation complete — {len(all_feat)} minutes for {sim_subject}")

        elif st.session_state.sim_log is not None:
            df_log = st.session_state.sim_log
            st.markdown('<div class="section-title">Last Simulation Results</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.pyplot(plot_timeseries(df_log,"heart_rate","Heart Rate (bpm)",ACCENT,"BPM"),
                      use_container_width=True)
            c2.pyplot(plot_timeseries(df_log,"stress_prob","Stress Probability",DANGER,"Probability"),
                      use_container_width=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Avg Stress Prob",   f"{df_log['stress_prob'].mean():.1%}")
            m2.metric("Avg Heart Rate",    f"{df_log['heart_rate'].mean():.1f} bpm")
            m3.metric("Peak Stress Prob",  f"{df_log['stress_prob'].max():.1%}")
            m4.metric("Minutes Monitored", len(df_log))
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#334155">
              Model loaded. Select a subject and click <b>▶ Run Live Simulation</b>.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if st.session_state.model is None:
        st.info("Train the model first using the sidebar.")
    else:
        rpt = st.session_state.report
        cm = st.session_state.cm
        r1 = rpt.get("1", {})
        ra = float(rpt.get("accuracy", 0))
        st.markdown('<div class="section-title">Classification Performance</div>',
                    unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Overall Accuracy",  f"{ra:.1%}")
        m2.metric("Stress Precision",  f"{r1.get('precision',0):.1%}")
        m3.metric("Stress Recall",     f"{r1.get('recall',0):.1%}")
        m4.metric("Stress F1-Score",   f"{r1.get('f1-score',0):.1%}")

        st.markdown('<div class="section-title">Visualisations</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_feature_importance(st.session_state.model),
                      use_container_width=True)
        with col2:
            st.pyplot(plot_confusion(cm), use_container_width=True)

        st.markdown('<div class="section-title">Subjects Loaded</div>',
                    unsafe_allow_html=True)
        st.write(", ".join(sorted(st.session_state.loaded_subjects)))
        st.markdown('<div class="section-title">Full Classification Report</div>',
                    unsafe_allow_html=True)
        df_rep = pd.DataFrame(rpt).T.dropna()
        st.dataframe(df_rep.style.format("{:.3f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Clinical Report
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.sim_log is None:
        st.info("Run a simulation first to generate the clinical report.")
    elif st.session_state.model is None or st.session_state.df_all is None:
        st.warning("Model data is no longer available. Please re-train the model using the sidebar, then run a simulation again.")
    else:
        df_log = st.session_state.sim_log
        agg    = st.session_state.df_all[FEATURES].median().to_dict()
        agg["heart_rate"] = df_log["heart_rate"].mean()
        agg["eda_mean"]   = df_log["eda_mean"].mean()

        fs, fc, fp = predict_minute(st.session_state.model,
                                    st.session_state.scaler, agg)
        st.markdown('<div class="section-title">Hourly AI Clinical Report</div>',
                    unsafe_allow_html=True)
        bcls = "badge-stress" if fs=="Distress Detected" else "badge-stable"
        cr1, cr2 = st.columns([2,1])
        with cr1:
            st.markdown(f"""
            <div style="margin-bottom:16px">
              <span class="badge {bcls}" style="font-size:0.9rem;padding:6px 18px">{fs}</span>
              <span style="font-family:'DM Mono',monospace;font-size:0.8rem;
                           color:#4A7FA5;margin-left:14px">
                Confidence: {fc:.1%}
              </span>
            </div>
            """, unsafe_allow_html=True)
        with cr2:
            st.pyplot(plot_risk_gauge(fp), use_container_width=True)

        rtext = generate_report(agg, fs, fc, len(df_log))
        st.markdown(f'<div class="report-box">{rtext}</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download Clinical Report (.txt)", data=rtext,
                           file_name=f"neurowatch_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB — Live Watch (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    if st.session_state.model is None:
        st.info("Train the model first using the sidebar — Live Watch predictions "
                 "reuse the same RandomForest + scaler as the other tabs.")
    else:
        st.markdown('<div class="section-title">Live Wearable Feed</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-banner">
          This tab polls a separate <code>api.py</code> process for streamed
          readings. Start it first — <code>uvicorn api:app --port 8000</code> —
          then feed it with <code>python watch_simulation.py</code> (or a real
          device posting to <code>/ingest</code>).
        </div>
        """, unsafe_allow_html=True)

        # ── API status ──────────────────────────────────────────────────
        status_ph = st.empty()

        def _render_status():
            try:
                r = requests.get(f"{st.session_state.live_api_url}/health", timeout=3)
                if r.status_code == 200:
                    h = r.json()
                    with status_ph.container():
                        s1, s2, s3 = st.columns(3)
                        s1.metric("API Status", "🟢 Online")
                        s2.metric("Buffered Readings", h.get("buffered_readings", 0))
                        s3.metric("Calibrated", "Yes" if h.get("calibrated") else "No")
                    return True
                status_ph.error(f"API returned {r.status_code}")
                return False
            except requests.exceptions.RequestException as e:
                status_ph.error(f"⚠️ Cannot reach API at {st.session_state.live_api_url} — "
                                 f"is `uvicorn api:app --port 8000` running? ({e})")
                return False

        api_online = _render_status()

        st.markdown('<div class="section-title">Live Monitor</div>',
                    unsafe_allow_html=True)

        if start_live:
            if not api_online:
                st.error("Cannot start — API is not reachable. Start api.py first.")
            else:
                log = list(st.session_state.live_log.to_dict("records")) \
                    if st.session_state.live_log is not None else []
                seen_ts = {r["timestamp"] for r in log} if log else set()

                prog_bar = st.progress(0, text="Listening for live readings…")
                gauge_ph = st.empty()
                c1, c2 = st.columns(2)
                hr_ph, ep_ph = c1.empty(), c2.empty()

                for cycle in range(poll_cycles):
                    try:
                        r = requests.get(f"{st.session_state.live_api_url}/latest", timeout=3)
                        if r.status_code == 200:
                            reading = r.json()
                            if reading["timestamp"] not in seen_ts:
                                seen_ts.add(reading["timestamp"])
                                feat = {k: reading[k] for k in FEATURES}
                                state, conf, sp = predict_minute(
                                    st.session_state.model, st.session_state.scaler, feat)
                                log.append({
                                    "minute": len(log) + 1,
                                    "timestamp": reading["timestamp"],
                                    "heart_rate": feat["heart_rate"],
                                    "eda_mean": feat["eda_mean"],
                                    "stress_prob": sp, "confidence": conf, "state": state,
                                })
                                df_log = pd.DataFrame(log)
                                st.session_state.live_log = df_log

                                with gauge_ph.container():
                                    gc = st.columns(5)
                                    gc[0].metric("Heart Rate", f"{feat['heart_rate']:.1f} bpm")
                                    gc[1].metric("EDA", f"{feat['eda_mean']:.3f} µS")
                                    gc[2].metric("Resp", f"{feat['resp_mean']:.3f}")
                                    gc[3].metric("Temp", f"{feat['temp_mean']:.1f} °C")
                                    gc[4].metric("SDNN", f"{feat['sdnn']:.4f} s")

                                    rg1, rg2 = st.columns([1, 2])
                                    with rg1:
                                        st.pyplot(plot_risk_gauge(sp), use_container_width=True)
                                    with rg2:
                                        bcls = "badge-stress" if state == "Distress Detected" else "badge-stable"
                                        st.markdown(f"""
                                        <div style="padding:20px 0">
                                          <div class="metric-label">Live AI Assessment</div>
                                          <div style="margin:10px 0">
                                            <span class="badge {bcls}">{state}</span>
                                          </div>
                                          <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                                      color:#4A7FA5;margin-top:8px">
                                            Confidence: <span style="color:#38BDF8">{conf:.1%}</span>
                                          </div>
                                          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                                                      color:#334155;margin-top:4px">
                                            Source reading: {reading['timestamp']}
                                          </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                with hr_ph:
                                    st.pyplot(plot_timeseries(df_log, "heart_rate",
                                                              "Heart Rate (bpm)", ACCENT, "BPM"),
                                              use_container_width=True)
                                with ep_ph:
                                    st.pyplot(plot_timeseries(df_log, "stress_prob",
                                                              "Stress Probability", DANGER, "Probability"),
                                              use_container_width=True)
                    except requests.exceptions.RequestException:
                        pass  # transient — keep polling

                    prog_bar.progress((cycle + 1) / poll_cycles,
                                      text=f"Poll {cycle + 1}/{poll_cycles}")
                    time.sleep(poll_seconds)

                prog_bar.empty()
                log_action(st.session_state.user_id, st.session_state.username,
                           "live_watch_poll", f"{poll_cycles} cycles @ {poll_seconds}s")
                st.success(f"✅ Polling complete — {len(log)} live readings received so far.")

        elif st.session_state.live_log is not None and len(st.session_state.live_log) > 0:
            df_log = st.session_state.live_log
            st.markdown('<div class="section-title">Last Live Session</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.pyplot(plot_timeseries(df_log, "heart_rate", "Heart Rate (bpm)", ACCENT, "BPM"),
                      use_container_width=True)
            c2.pyplot(plot_timeseries(df_log, "stress_prob", "Stress Probability", DANGER, "Probability"),
                      use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Stress Prob", f"{df_log['stress_prob'].mean():.1%}")
            m2.metric("Avg Heart Rate", f"{df_log['heart_rate'].mean():.1f} bpm")
            m3.metric("Peak Stress Prob", f"{df_log['stress_prob'].max():.1%}")
            m4.metric("Readings Received", len(df_log))
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#334155">
              No live readings yet. Start <code>api.py</code> and
              <code>watch_simulation.py</code>, then click <b>▶ Start Live Feed</b>.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    **NeuroWatch** is a Human–AI hybrid decision-support system for detecting emotional distress
    in bedridden patients who may not be able to communicate verbally.

    The AI continuously analyzes passive physiological signals and generates interpretable clinical
    summaries to assist doctors — it does **not** replace clinical judgment.

    ---
    **Signals Used (WESAD Chest Sensor)**

    | Signal | Sampling Rate | Features Extracted |
    |---|---|---|
    | ECG | 700 Hz | Heart rate, SDNN, RMSSD |
    | EDA | 4 Hz | Mean, Std, Peaks, Arousal Index |
    | Respiration | 700 Hz | Mean, Std |
    | Temperature | 700 Hz | Mean |

    ---
    **Architecture**
    - **Window size**: 60-second segments
    - **Model**: Random Forest (600 trees, balanced class weights)
    - **Scaler**: StandardScaler (fit on training set)
    - **Auth**: SQLite user database with PBKDF2-HMAC-SHA256 password hashing
    - **Live Watch (Phase 2)**: `api.py` (FastAPI) buffers streamed readings from
      `watch_simulation.py` — this app polls it and runs the *same* trained
      model + scaler, so Live Watch predictions stay consistent with the
      WESAD playback and evaluation tabs.

    ---
    **Human–AI Hybrid Concept**
    ```
    Physiological Signals
          ↓
    AI Feature Extraction (ECG · EDA · Resp · Temp)
          ↓
    Random Forest Classifier
          ↓
    Confidence + Risk Score
          ↓
    Clinical Report ──→ Clinician Review ──→ Final Decision
    ```
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Admin Panel (admin role only)
# ─────────────────────────────────────────────────────────────────────────────
if tab_admin:
    with tab_admin:
        st.markdown('<div class="section-title">User Management</div>',
                    unsafe_allow_html=True)

        users = get_all_users()
        if users:
            df_users = pd.DataFrame([dict(u) for u in users])
            df_users["last_login"] = df_users["last_login"].fillna("Never")
            st.dataframe(df_users[["id","full_name","email","username","role",
                                   "created_at","last_login"]],
                         use_container_width=True)
        else:
            st.info("No users found.")

        st.markdown('<div class="section-title">Delete User</div>',
                    unsafe_allow_html=True)
        col_d1, col_d2 = st.columns([2,1])
        with col_d1:
            del_id = st.number_input("User ID to delete", min_value=1, step=1,
                                     key="del_uid")
        with col_d2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Delete User", type="primary"):
                if del_id == st.session_state.user_id:
                    st.error("You cannot delete your own account.")
                else:
                    delete_user(int(del_id))
                    log_action(st.session_state.user_id,
                               st.session_state.username,
                               "delete_user", f"deleted user_id={del_id}")
                    st.success(f"User {del_id} deleted.")
                    st.rerun()

        st.markdown('<div class="section-title">Session Activity Log</div>',
                    unsafe_allow_html=True)
        logs = get_session_log(limit=100)
        if logs:
            df_actlog = pd.DataFrame([dict(l) for l in logs])
            st.dataframe(df_actlog, use_container_width=True)
        else:
            st.info("No activity logged yet.")
