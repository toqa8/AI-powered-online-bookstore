# this code generates description, difficulty level, target audience, and themes for book 1 and book 2.

import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
# This is where GROQ_API_KEY is stored securely
load_dotenv()

# Initialize Groq client (LLM provider)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Paths configuration
BOOKS_PATH = "data/books.json"
CONTENT_DIR = "data/books_content"


# ---------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------

def load_books():
    """
    Load books metadata from JSON file.
    This is our main dataset that will be updated
    with AI-generated fields (description, themes, etc.)
    """
    with open(BOOKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_books(books):
    """
    Save updated books data back to JSON file.
    """
    with open(BOOKS_PATH, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=2, ensure_ascii=False)


def load_book_content(content_file):
    """
    Load raw book text from .txt file.
    """
    path = os.path.join(CONTENT_DIR, content_file)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------
# Chunking Logic
# ---------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split long book text into smaller overlapping chunks.

    Why chunking?
    - LLMs have context limits
    - Helps select relevant parts of the book
    - Improves quality of metadata generation

    overlap ensures continuity between chunks.
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def select_representative_chunks(chunks):
    """
    Instead of sending ALL chunks to the LLM,
    we select representative parts of the book.

    Why?
    - Reduce token cost
    - Avoid overwhelming the model
    - Still preserve global structure of the book

    Strategy used here:
    - First chunk (introduction)
    - Middle chunk (development)
    - Last chunk (conclusion)
    """
    if len(chunks) <= 3:
        return chunks

    return [
        chunks[0],
        chunks[len(chunks) // 2],
        chunks[-1]
    ]


# ---------------------------------------------------
# LLM Prompt Engineering
# ---------------------------------------------------

def build_prompt(book_title, chunks):
    """
    Build structured prompt for metadata generation.
    We enforce JSON output for structured parsing later.
    """

    context = "\n\n".join(chunks)

    return f"""
You are a professional book analyst.

Analyze the following book excerpts and generate metadata.

BOOK TITLE: {book_title}

EXCERPTS:
{context}

Return ONLY valid JSON in this format:

{{
  "description": "2-3 sentence book description",
  "difficulty": "Beginner | Intermediate | Advanced",
  "audience": "Who this book is for",
  "themes": ["theme1", "theme2", "theme3"]
}}

Rules:
- Do NOT copy text directly
- Be concise and accurate
- No extra text outside JSON
- If the book is NOT academic or educational:
  → set "is_academic": false
  → set "difficulty": null
- If the book IS academic/educational:
  → set "is_academic": true
  → assign difficulty based on complexity
- Do NOT guess difficulty for novels or fiction
"""


def generate_metadata(book_title, chunks):
    """
    Send prompt to Groq LLM and parse the response.

    Steps:
    1. Build prompt
    2. Call LLM API
    3. Extract response
    4. Parse JSON output
    """

    prompt = build_prompt(book_title, chunks)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content

    # Try parsing model output into JSON
    # If parsing fails, print raw output for debugging
    try:
        return json.loads(content)
    except:
        print("❌ Failed to parse JSON response")
        print("Raw output:\n", content)
        return None


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run():

    print(1)
    """
    Main execution pipeline:

    FULL FLOW:
    1. Load books.json
    2. For each book:
        a. Load book content (.txt)
        b. Chunk the content
        c. Select representative chunks
        d. Send to LLM for metadata generation
        e. Update book fields in JSON
    3. Save updated dataset
    """

    books = load_books()
    print(f"📦 Loaded books: {len(books)}")
    print("📚 Books data:", books)

    for book in books:
        print(f"\n📚 Processing: {book['title']}")

        content_file = book.get("content_file")

        # Skip books that do not have text content
        if not content_file:
            print("⏭️ No content file found, skipping...")
            continue

        # Step 1: Load raw book text
        text = load_book_content(content_file)

        # Step 2: Convert text into chunks
        chunks = chunk_text(text)

        # Step 3: Select representative chunks (PoC optimization)
        selected_chunks = select_representative_chunks(chunks)

        # Step 4: Generate metadata using LLM
        metadata = generate_metadata(book["title"], selected_chunks)

        if not metadata:
            print("❌ Metadata generation failed")
            continue

        # Step 5: Update book object with AI-generated fields
        book["description"] = metadata.get("description", "")
        book["difficulty"] = metadata.get("difficulty", "")
        book["audience"] = metadata.get("audience", "")
        book["themes"] = metadata.get("themes", [])

        print("✅ Metadata generated successfully")

    # Step 6: Save updated dataset
    save_books(books)
    print("\n🎉 All books updated successfully!")


# ---------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    run()


"""
========================================================
📌 OVERALL PIPELINE SUMMARY

This script performs OFFLINE AI processing:

Book Content (.txt)
        ↓
Chunking (split long text)
        ↓
Select representative chunks (PoC optimization)
        ↓
Send to LLM (Groq)
        ↓
Generate structured metadata:
    - Description
    - Difficulty
    - Audience
    - Themes
        ↓
Update books.json (acts as lightweight database)
        ↓
Used later by:
    - Semantic Search
    - Recommendations
    - Chatbot
    - Mind Maps (indirectly via context)
========================================================
"""