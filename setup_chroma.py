import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# 📌 Paths setup (important)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOOKS_PATH = os.path.join(BASE_DIR, "data", "books.json")
CONTENT_DIR = os.path.join(BASE_DIR, "data", "books_content")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# =========================
# 📌 Load embedding model
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📌 Init ChromaDB
# =========================

client = chromadb.PersistentClient(path=CHROMA_PATH)

books_collection = client.get_or_create_collection("books")
chunks_collection = client.get_or_create_collection("chunks")

# =========================
# 📌 Load books data
# =========================

with open(BOOKS_PATH, "r", encoding="utf-8") as f:
    books = json.load(f)

print(f"📚 Loaded {len(books)} books")

# =========================
# 📌 Helper: chunking function
# =========================

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks for better RAG retrieval
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks

# =========================
# 📌 1. Embed books metadata
# =========================

print("\n🔵 Embedding books metadata...")

for book in books:

    # Combine structured metadata into one text
    metadata_text = f"""
    Title: {book['title']}
    Author: {book['author']}
    Genres: {' '.join(book.get('genres', []))}
    Description: {book.get('description', '')}
    Themes: {' '.join(book.get('themes', []))}
    Audience: {' '.join(book.get('audience', [])) if book.get('audience') else ''}
    """

    embedding = model.encode(metadata_text).tolist()

    books_collection.add(
        ids=[book["id"]],
        embeddings=[embedding],
        documents=[metadata_text],
        metadatas=[{
            "id": book["id"],
            "title": book["title"],
            "author": book["author"],
            "genres": book.get("genres", []),
            "rating": book.get("rating", 0),
            "price": book.get("price", 0)
        }]
    )

print("✅ Books metadata embedded")

# =========================
# 📌 2. Embed book content chunks
# =========================

print("\n🟣 Embedding book content chunks...")

for book in books:

    content_file = book.get("content_file")

    if not content_file:
        print(f"⏭️ Skipping {book['title']} (no content)")
        continue

    content_path = os.path.join(CONTENT_DIR, content_file)

    if not os.path.exists(content_path):
        print(f"❌ Missing file: {content_path}")
        continue

    with open(content_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    print(f"📖 {book['title']} → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        chunks_collection.add(
            ids=[f"{book['id']}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "book_id": book["id"],
                "chunk_index": i
            }]
        )

print("\n🎉 ChromaDB setup completed successfully!")