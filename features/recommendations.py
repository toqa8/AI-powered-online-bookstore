import numpy as np
from sentence_transformers import SentenceTransformer

# =============================
# Load embedding model
# =============================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================
# Cosine similarity
# =============================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================
# Build user vector
# =============================
def build_user_vector(user_profile, books, book_embeddings):
    """
    User representation = 
    70% reading history embeddings + 30% preferences embedding
    """

    history_ids = set(user_profile.get("reading_history", []))

    # -------------------------
    # 1. History embeddings
    # -------------------------
    history_vectors = []

    for i, book in enumerate(books):
        if book["id"] in history_ids:
            history_vectors.append(book_embeddings[i])

    history_vector = None
    if len(history_vectors) > 0:
        history_vector = np.mean(history_vectors, axis=0)

    # -------------------------
    # 2. Preferences embedding
    # -------------------------
    pref_text = f"""
    Preferred genres: {' '.join(user_profile.get('preferred_genres', []))}
    Preferred authors: {' '.join(user_profile.get('preferred_authors', []))}
    """

    pref_vector = model.encode(pref_text)

    # -------------------------
    # 3. Combine
    # -------------------------
    if history_vector is not None:
        user_vector = 0.7 * history_vector + 0.3 * pref_vector
    else:
        user_vector = pref_vector

    return user_vector


# =============================
# Recommendation engine
# =============================
def get_recommendations(user_profile, books, embeddings, top_k=3):

    # Build user vector inside (clean API for Streamlit)
    user_vector = build_user_vector(user_profile, books, embeddings)

    history_ids = set(user_profile.get("reading_history", []))

    scored = []

    # -------------------------
    # Score all books
    # -------------------------
    for i, book in enumerate(books):

        # skip already read
        if book["id"] in history_ids:
            continue

        score = cosine(user_vector, embeddings[i])
        scored.append((book, score))

    # -------------------------
    # Sort by score
    # -------------------------
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]