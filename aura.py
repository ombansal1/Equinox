
# aura.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re

# --- Load model once globally for efficiency ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Keyword dictionaries for topic detection ---
TOPIC_KEYWORDS = {
    "work": ["job", "office", "manager", "project", "work"],
    "relationships": ["friend", "love", "partner", "family", "relationship"],
    "health": ["doctor", "tired", "sleep", "health", "pain", "exercise"],
    "study": ["school", "college", "study", "exam", "assignment"],
    "gratitude": ["thank", "grateful", "blessed", "appreciate"],
    "self-image": ["confidence", "anxiety", "feel", "myself", "mental"]
}

AURA_MAP = {
    0: ("🌿 Calm Green", "Reflective and grounded — you often express thoughtfulness."),
    1: ("🔥 Radiant Orange", "Energetic and expressive — your posts show high engagement."),
    2: ("🌊 Tranquil Blue", "Balanced and introspective — calm tone with positive reflections."),
    3: ("🌪️ Stormy Gray", "You’ve shared signs of stress or emotional intensity recently."),
    4: ("🌸 Blossom Pink", "Compassionate and emotionally aware — empathetic tone detected."),
    5: ("🌞 Bright Yellow", "Optimistic and uplifting — your tone reflects positivity.")
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    return text

def topic_distribution(texts):
    counts = {t: 0 for t in TOPIC_KEYWORDS}
    total = 0
    for t in texts:
        total += 1
        text = preprocess(t)
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in text for k in keywords):
                counts[topic] += 1
    if total == 0:
        return counts
    return {k: round(v / total * 100, 1) for k, v in counts.items()}

def analyze_aura(posts):
    """Given list of posts, returns aura, description, and topic insight."""
    if not posts:
        return None

    texts = [p["text"] for p in posts if p["text"].strip()]
    if len(texts) < 1:
        return {
    "aura": "🌊 Tranquil Blue",
    "description": "Balanced and introspective — calm tone with positive reflections.",
    "topics": {},
    "comment": "Not enough posts for a full analysis yet."
}


    embeddings = model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    user_cluster = int(np.argmax(np.bincount(clusters)))
    aura_color, aura_desc = AURA_MAP.get(user_cluster, ("🌊 Tranquil Blue", "Balanced mood."))

    topics = topic_distribution(texts)
    top_topic = max(topics, key=topics.get)
    comment = f"You talk about {top_topic} {topics[top_topic]}% of the time."

    return {
        "aura": aura_color,
        "description": aura_desc,
        "topics": topics,
        "comment": comment
    }
