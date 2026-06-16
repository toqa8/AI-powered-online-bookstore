import os
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# =========================
# Load embedding model
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Connect to Chroma
# =========================

client = chromadb.PersistentClient(path=CHROMA_PATH)

books_collection = client.get_collection("books")


# =========================
# Semantic Search Function
# =========================

def semantic_search(query, top_k=2):

    query_embedding = model.encode(query).tolist()

    results = books_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    books = []

    for metadata in results["metadatas"][0]:

        books.append({
            "title": metadata["title"],
            "author": metadata["author"],
            "genres": metadata["genres"],
            "rating": metadata["rating"],
            "price": metadata["price"]
        })

    return books


# =========================
# Test
# =========================

if __name__ == "__main__":

    query = input("Search: ")

    results = semantic_search(query)

    print("\nResults:\n")

    for i, book in enumerate(results, start=1):

        print(f"{i}. {book['title']}")
        print(f"   Author: {book['author']}")
        print(f"   Genres: {book['genres']}")
        print(f"   Rating: {book['rating']}")
        print()