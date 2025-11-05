# app.py
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ingest import scrape_subreddits, load_cached_df
from aura import analyze_aura
from nlp_bert import analyze_emotions

# --- CONFIG ---
REDDIT_CLIENT_ID = "M1I1jvc74xBb2xr-QuK2zQ"
REDDIT_CLIENT_SECRET = "qSFZqF4lbrk5kZ9q3UP44n0RMBHrig"
REDDIT_USER_AGENT = "wellness-tracker-demo"

# --- APP INIT ---
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
user_posts = {}

# --- Reddit Init ---
def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

# --- Text Preprocessing ---
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Fetch Reddit Posts for a Specific User ---
def fetch_user_submissions(username, limit=100):
    reddit = init_reddit()
    ruser = reddit.redditor(username)
    posts = []
    for s in ruser.submissions.new(limit=limit):
        text = (s.title or "") + " " + (s.selftext or "")
        clean_text = preprocess_text(text)
        posts.append({
            "id": s.id,
            "title": s.title,
            "created": datetime.utcfromtimestamp(s.created_utc),
            "text": clean_text,
        })
    user_posts[username] = posts
    return posts

# --- Sentiment Analysis ---
def analyze_posts(posts):
    results = []
    for p in posts:
        scores = analyzer.polarity_scores(p["text"])
        results.append({
            "id": p["id"],
            "title": p["title"],
            "created": p["created"],
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        })
    return results

# --- Aggregate Daily Mood ---
def get_daily_mood(username, days=60):
    posts = user_posts.get(username, [])
    if not posts:
        return []
    analyzed = analyze_posts(posts)
    cutoff = datetime.utcnow() - timedelta(days=days)
    buckets = {}
    for a in analyzed:
        if a["created"] < cutoff:
            continue
        d = a["created"].date().isoformat()
        buckets.setdefault(d, []).append(a["compound"])
    trend = [
        {"date": d, "avg_compound": sum(vals)/len(vals)}
        for d, vals in sorted(buckets.items())
    ]
    return trend

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    username = request.args.get("username")
    if not username:
        return redirect(url_for("home"))
    return render_template("dashboard.html", username=username)

# --- Therapist Pages ---

@app.route("/therapist")
def therapist_dashboard():
    """Main Therapist Dashboard"""
    return render_template("therapist_dashboard.html")

@app.route("/patient/<string:author>")
def patient_detail(author):
    """Detailed page for one patient"""
    df = load_cached_df()
    posts = df[df["author"] == author].to_dict("records")
    aura = analyze_aura([{"text": p["title"] + " " + p.get("selftext", "")} for p in posts[:30]])
    emotions = analyze_emotions([{"text": p["title"] + " " + p.get("selftext", "")} for p in posts[:20]])
    return render_template("patient_detail.html", author=author, aura=aura, emotions=emotions)

@app.route("/api/therapist/search")
def therapist_search():
    """Dynamic search and filter API"""
    name_query = request.args.get("name", "").lower()
    emotion_filter = request.args.get("emotion", "").lower()

    df = load_cached_df()
    if df.empty:
        return jsonify({"patients": [], "note": "Cache empty. Please scrape first."})

    # Aggregate per author
    patients = (
        df.groupby("author")
        .agg({"title": "count"})
        .rename(columns={"title": "post_count"})
        .reset_index()
    )
    patients["dominant_emotion"] = [
        "happy" if i % 3 == 0 else "sad" if i % 3 == 1 else "calm" for i in range(len(patients))
    ]  # dummy placeholder

    if name_query:
        patients = patients[patients["author"].str.lower().str.contains(name_query)]
    if emotion_filter:
        patients = patients[patients["dominant_emotion"].str.lower() == emotion_filter]

    result = patients.head(50).to_dict("records")
    return jsonify({"patients": result, "count": len(result)})

# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=True)
