auth.py

Simple username/password authentication with Sign Up support for the
Student Performance Intelligence app.

Users are stored in a local CSV file (data/users.csv) with salted,
hashed passwords (PBKDF2-HMAC-SHA256, Python standard library only —
no extra dependency required). This is a lightweight solution suitable
for a single-app demo / academic project.

Note: on Streamlit Community Cloud the filesystem resets whenever the
app is rebooted or redeployed, so registered accounts will reset at
that point too. For a persistent multi-session deployment, swap the
CSV storage in this file for a real database.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets

import pandas as pd
import streamlit as st

USERS_FILE = os.path.join("data", "users.csv")
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,20}$")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------
def _ensure_users_file() -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=["name", "username", "salt", "password_hash"]).to_csv(
            USERS_FILE, index=False
        )


def _load_users() -> pd.DataFrame:
    _ensure_users_file()
    return pd.read_csv(USERS_FILE, dtype=str).fillna("")


def _save_users(df: pd.DataFrame) -> None:
    _ensure_users_file()
    df.to_csv(USERS_FILE, index=False)


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, stdlib only)
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


# ---------------------------------------------------------------------------
# Public account operations
# ---------------------------------------------------------------------------
def username_exists(username: str) -> bool:
    users = _load_users()
    if users.empty:
        return False
    return (users["username"].str.lower() == username.lower()).any()


def create_user(name: str, username: str, password: str) -> tuple[bool, str]:
    name = name.strip()
    username = username.strip()

    if not name:
        return False, "Please enter your name."
    if not USERNAME_PATTERN.match(username):
        return False, "Username must be 3-20 characters: letters, numbers, underscore only."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if username_exists(username):
        return False, "This username is already taken. Please choose another."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    users = _load_users()
    new_row = pd.DataFrame([{
        "name": name,
        "username": username,
        "salt": salt,
        "password_hash": password_hash,
    }])
    users = pd.concat([users, new_row], ignore_index=True)
    _save_users(users)
    return True, "Account created successfully. You can now log in from the 'Log In' tab."


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    users = _load_users()
    if users.empty:
        return False, ""

    match = users[users["username"].str.lower() == username.strip().lower()]
    if match.empty:
        return False, ""

    row = match.iloc[0]
    if _verify_password(password, row["salt"], row["password_hash"]):
        return True, row["name"]
    return False, ""


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        ok, name = authenticate_user(username, password)
        if ok:
            st.session_state.authenticated = True
            st.session_state.user_name = name
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


def _signup_form() -> None:
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up", type="primary", use_container_width=True)

    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
        ok, message = create_user(name, username, password)
        if ok:
            st.success(message)
        else:
            st.error(message)


def login_signup_gate() -> bool:
    """Render the login / sign up UI and gate the rest of the app.

    Returns True if the current session is authenticated. When False,
    the caller should stop rendering the rest of the app (st.stop()).
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        """
        <div style="text-align:center; margin-top:1rem; margin-bottom:1.5rem;">
            <h1 style="margin-bottom:0.2rem;">Student Performance Intelligence</h1>
            <p style="color:#5c6b7a;">Please log in or create an account to continue</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.2, 1])
    with center:
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])
        with tab_login:
            _login_form()
        with tab_signup:
            _signup_form()

    return False


def render_logout_sidebar() -> None:
    """Show the logged-in user's name and a logout button in the sidebar."""
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.get('user_name', '')}")
    if st.sidebar.button("Logout", use_container_width=True):
        for key in ("authenticated", "user_name", "username"):
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.markdown("---")
