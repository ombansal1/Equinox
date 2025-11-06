# personality.py
"""
Ultra-light Big Five estimator (0–100) derived from language/emotion signals.
This is a heuristic (explainable) stand-in — not a clinical instrument.
"""

from collections import Counter
import re

def _scale(x, lo=0.0, hi=1.0):
    # map [lo,hi] to [0,100]
    x = max(lo, min(hi, x))
    return int(round(100 * (x - lo) / (hi - lo))) if hi > lo else 0

def _lex_score(texts):
    text = " ".join(texts).lower()
    def count(ws): return sum(1 for w in ws if re.search(rf"\b{re.escape(w)}\b", text))
    # tiny lexicons
    pos_words = ["grateful","curious","excited","learn","explore","create","together","helpful"]
    neg_words = ["tired","alone","hopeless","angry","guilty","worthless","fail","panic"]
    social    = ["friends","party","talk","team","community","meet","hangout","club"]
    planful   = ["schedule","plan","routine","goal","deadline","organize","checklist","task"]
    kind      = ["support","empathy","kind","care","thanks","sorry","appreciate","help"]
    worry     = ["anxious","anxiety","worry","overthink","stressed","panic","afraid","nervous"]

    L = len(text.split())
    L = max(L, 1)

    return {
        "pos": count(pos_words)/L,
        "neg": count(neg_words)/L,
        "social": count(social)/L,
        "planful": count(planful)/L,
        "kind": count(kind)/L,
        "worry": count(worry)/L,
    }

def analyze_big5(texts):
    s = _lex_score(texts)
    # Openness: curiosity/novelty language
    openness = _scale(0.6*s["pos"] + 0.4*s["social"], 0, 0.02)
    # Conscientiousness: planning, routine
    conscientiousness = _scale(0.8*s["planful"] - 0.2*s["neg"], -0.005, 0.02)
    # Extraversion: social words and positive affect
    extraversion = _scale(0.7*s["social"] + 0.3*s["pos"], 0, 0.02)
    # Agreeableness: prosocial language
    agreeableness = _scale(0.8*s["kind"] - 0.2*s["neg"], -0.005, 0.02)
    # Neuroticism: worry/negative affect
    neuroticism = _scale(0.7*s["worry"] + 0.3*s["neg"], 0, 0.02)

    # clamp to [0,100]
    def clamp(v): return max(0, min(100, v))
    return {
        "openness": clamp(openness),
        "conscientiousness": clamp(conscientiousness),
        "extraversion": clamp(extraversion),
        "agreeableness": clamp(agreeableness),
        "neuroticism": clamp(neuroticism),
    }
