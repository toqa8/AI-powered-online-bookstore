# features/why_recommended.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_why_recommended(user_profile, book):
    """
    user_profile: dict (reading_history, preferred_genres, preferred_authors)
    book: dict (title, author, genres, themes, audience)
    returns: string explanation
    """

    # Handle audience whether it's a list or string
    audience = book.get("audience", "")
    if isinstance(audience, list):
        audience = ", ".join(audience)

    prompt = f"""
You are a book recommendation assistant.

User profile:
- Name: {user_profile.get('name', 'the user')}
- Favorite genres: {', '.join(user_profile.get('preferred_genres', []))}
- Favorite authors: {', '.join(user_profile.get('preferred_authors', []))}

Book being recommended:
- Title: "{book['title']}"
- Author: {book['author']}
- Genres: {', '.join(book.get('genres', []))}
- Themes: {', '.join(book.get('themes', []))}
- Audience: {audience}
- Description: {book.get('description', '')}

Write a very short and concise (2 sentences MAX) explaining why this specific book suits this specific user.

Rules:
- Mention at least one concrete connection between the user's interests and the book
- Do NOT use the phrase "based on your preferences"
- Do NOT use the phrase "I think" or "I believe"
- Sound natural and personal
- 2 sentences only
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()