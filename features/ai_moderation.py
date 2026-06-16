import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# STEP 1: classify toxicity
# =========================

def check_toxicity(review):
    prompt = f"""
You are a moderation system.

Check if the review contains:
- insults
- offensive language
- abusive words

IMPORTANT:
- Mild negative opinions like "boring", "slow", "not interesting" are NOT toxic.

Return ONLY:
- safe
- toxic

Review:
"{review}"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# =========================
# STEP 2: filter reviews
# =========================

def filter_reviews(reviews):
    safe_reviews = []
    removed_reviews = []

    for r in reviews:
        label = check_toxicity(r)

        if "toxic" in label.lower():
            removed_reviews.append(r)
        else:
            safe_reviews.append(r)

    return safe_reviews, removed_reviews