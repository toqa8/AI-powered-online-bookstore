# app.py

import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from features.recommendations import get_recommendations
from features.why_recommended import get_why_recommended
from features.semantic_search import semantic_search
from features.ai_moderation import filter_reviews

import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build ChromaDB if not exists or empty
chroma_path = os.path.join(BASE_DIR, "chroma_db")
marker = os.path.join(BASE_DIR, "chroma_db", ".setup_done")

if not os.path.exists(marker):
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "setup_chroma.py")], check=True)
    open(marker, "w").close()

load_dotenv()

# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Bookstore AI",
    page_icon="📚",
    layout="wide"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #faf7f2;
        color: #2c2c2c;
    }

    .main { background-color: #faf7f2; }

    /* Search bar */
    .search-wrapper {
        background: #ffffff;
        border: 1px solid #e0d8cc;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 2rem;
    }

    /* Section titles */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #8b6914;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0d8cc;
        padding-bottom: 0.5rem;
    }

    /* Book card */
    .book-card {
        background: #ffffff;
        border: 1px solid #e0d8cc;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }

    .book-card:hover {
        border-color: #8b6914;
    }

    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #8b6914;
        margin-bottom: 0.2rem;
    }

    .book-author {
        font-size: 0.82rem;
        color: #888;
        margin-bottom: 0.5rem;
    }

    .book-meta {
        font-size: 0.78rem;
        color: #666;
    }

    .genre-tag {
        display: inline-block;
        background: #f0e8d8;
        color: #8b6914;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    /* Profile card */
    .profile-card {
        background: #ffffff;
        border: 1px solid #e0d8cc;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .profile-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #8b6914;
        margin-bottom: 0.8rem;
    }

    .profile-label {
        font-size: 0.72rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .profile-value {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.8rem;
    }

    /* Review tags */
    .safe-badge {
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 4px;
    }

    .unsafe-badge {
        background: #ffebee;
        color: #c62828;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 4px;
    }

    .review-item {
        background: #fafafa;
        border-left: 3px solid #e0d8cc;
        padding: 0.6rem 0.8rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #444;
    }

    .review-safe { border-left-color: #4caf50; }
    .review-unsafe { border-left-color: #ef5350; }

    /* Why recommended box */
    .why-box {
        background: #fffbf4;
        border: 1px solid #e8d8a0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.83rem;
        color: #555;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: transparent;
        border: 1px solid #e0d8cc;
        color: #8b6914;
        border-radius: 8px;
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
    }

    .stButton > button:hover {
        border-color: #8b6914;
        background: #faf7f2;
    }

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e0d8cc;
        border-radius: 12px;
    }

    .stTextInput > div > div > input {
        background: #ffffff;
        border: 1px solid #e0d8cc;
        color: #2c2c2c;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load data
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    with open(os.path.join(BASE_DIR, "data", "books.json"), "r", encoding="utf-8") as f:
        books = json.load(f)
    with open(os.path.join(BASE_DIR, "data", "user_profile.json"), "r", encoding="utf-8") as f:
        user = json.load(f)
    with open(os.path.join(BASE_DIR, "data", "reviews.json"), "r", encoding="utf-8") as f:
        reviews = json.load(f)
    return books, user, reviews

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

books, user_profile, reviews = load_data()
model = load_model()

# =========================
# Precompute book embeddings
# =========================

@st.cache_data
def get_book_embeddings(books):
    _model = load_model()
    embeddings = []
    for book in books:
        text = f"{book['title']} {book['author']} {' '.join(book.get('genres', []))} {book.get('description', '')} {' '.join(book.get('themes', []))}"
        embeddings.append(_model.encode(text))
    return np.array(embeddings)

book_embeddings = get_book_embeddings(books)

# =========================
# Session state
# =========================

if "why_cache" not in st.session_state:
    st.session_state.why_cache = {}

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# =========================
# Layout: 3 columns
# =========================

left_col, main_col, right_col = st.columns([1.2, 2.5, 1.5])

# =========================================
# LEFT: User Profile
# =========================================

with left_col:
    st.markdown('<div class="section-title">👤 Profile</div>', unsafe_allow_html=True)

    genres_str = ", ".join(user_profile.get("preferred_genres", []))
    authors_str = ", ".join(user_profile.get("preferred_authors", []))
    history_titles = [b["title"] for b in books if b["id"] in user_profile.get("reading_history", [])]

    st.markdown(f'<div class="profile-card"><div class="profile-name"> {user_profile["name"]}</div></div>', unsafe_allow_html=True)

    st.markdown("**Favourite Genres**")
    st.caption(genres_str)

    st.markdown("**Favourite Authors**")
    st.caption(authors_str)

    st.markdown("**Reading History**")
    for title in history_titles:
        st.caption(f"• {title}")

# =========================================
# MAIN: Search + Recommendations + Reviews
# =========================================

with main_col:

    # --- Semantic Search ---
    st.markdown('<div class="section-title">🔍 Search Books</div>', unsafe_allow_html=True)

    query = st.text_input(
        label="search",
        placeholder='Try "a story about identity and imagination" ...',
        label_visibility="collapsed"
    )

    if query and query != st.session_state.search_query:
        st.session_state.search_query = query
        st.session_state.search_results = semantic_search(query, top_k=3)

    if st.session_state.search_results:
        st.markdown("**Search Results:**")
        for result in st.session_state.search_results:
            genres_display = result.get("genres", "")
            if isinstance(genres_display, list):
                genres_display = ", ".join(genres_display)
            st.markdown(f"""
            <div class="book-card">
                <div class="book-title">{result['title']}</div>
                <div class="book-author">{result['author']}</div>
                <div class="book-meta">⭐ {result['rating']} &nbsp;|&nbsp; 💰 {result['price']} EGP &nbsp;|&nbsp; {genres_display}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Recommended Books ---
    st.markdown('<div class="section-title">✨ Recommended For You</div>', unsafe_allow_html=True)

    recommendations = get_recommendations(user_profile, books, book_embeddings, top_k=4)

    for book, score in recommendations:
        book_id = book["id"]

        genres_tags = "".join([f'<span class="genre-tag">{g}</span>' for g in book.get("genres", [])])
        audience = book.get("audience", "")
        if isinstance(audience, list):
            audience = ", ".join(audience)

        themes_str = ", ".join(book.get("themes", []))

        st.markdown(f"""
        <div class="book-card">
            <div class="book-title">{book['title']}</div>
            <div class="book-author">by {book['author']}</div>
            <div style="margin: 0.4rem 0">{genres_tags}</div>
            <div class="book-meta">⭐ {book['rating']} &nbsp;|&nbsp; 💰 {book['price']} EGP</div>
            <div class="book-meta">🎯 {audience}</div>
            <div class="book-meta">🏷️ {themes_str}</div>
        </div>
        """, unsafe_allow_html=True)

        col_btn, col_space = st.columns([1, 4])
        with col_btn:
            if st.button(f"💡 Why this?", key=f"why_{book_id}"):
                if book_id not in st.session_state.why_cache:
                    with st.spinner("Thinking..."):
                        explanation = get_why_recommended(user_profile, book)
                        st.session_state.why_cache[book_id] = explanation

        if book_id in st.session_state.why_cache:
            st.markdown(f"""
            <div class="why-box">
                💬 {st.session_state.why_cache[book_id]}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# RIGHT: All Books + Reviews
# =========================================

with right_col:
    st.markdown('<div class="section-title">📚 All Books</div>', unsafe_allow_html=True)

    for book in books:
        book_id = book["id"]
        has_reviews = book_id in reviews

        label = f"{'📘'} {book['title']}"

        with st.expander(label):
            st.markdown(f"**{book['author']}**")
            st.markdown(f"⭐ {book['rating']} | 💰 {book['price']} EGP")

            genres_display = ", ".join(book.get("genres", []))
            st.markdown(f"*{genres_display}*")

            st.markdown(f"{book.get('description', '')}")

            audience = book.get("audience", "")
            if isinstance(audience, list):
                audience = ", ".join(audience)
            themes_str = ", ".join(book.get("themes", []))

            if audience:
                st.markdown(f"🎯 *Audience:* {audience}")
            if themes_str:
                st.markdown(f"🏷️ *Themes:* {themes_str}")

            if has_reviews:
                st.markdown("---")
                st.markdown("**Reviews**")

                book_reviews = reviews[book_id]

                if st.button("🛡️ Analyse Reviews", key=f"mod_{book_id}"):
                    with st.spinner("Analysing..."):
                        safe, unsafe = filter_reviews(book_reviews)
                        st.session_state[f"reviews_{book_id}"] = {
                            "safe": safe,
                            "unsafe": unsafe
                        }

                if f"reviews_{book_id}" in st.session_state:
                    result = st.session_state[f"reviews_{book_id}"]

                    st.markdown(f'<span class="safe-badge">✅ Safe ({len(result["safe"])})</span>', unsafe_allow_html=True)
                    for r in result["safe"]:
                        st.markdown(f'<div class="review-item review-safe">{r}</div>', unsafe_allow_html=True)

                    st.markdown(f'<span class="unsafe-badge">🚫 Removed ({len(result["unsafe"])})</span>', unsafe_allow_html=True)
                    for r in result["unsafe"]:
                        st.markdown(f'<div class="review-item review-unsafe">{r}</div>', unsafe_allow_html=True)
            else:
                st.markdown("*No reviews yet.*")