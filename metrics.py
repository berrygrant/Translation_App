# metrics.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

smooth = SmoothingFunction().method1

def compute_grammaticality(sentence):
    verbs = ["is", "are", "was", "were", "has", "have", "do", "does", "did",
             "go", "went", "see", "saw", "says", "said", "run", "runs"]
    score = 0.0
    if sentence and sentence[0].isupper():
        score += 0.3
    if sentence and sentence[-1] in ".!?":
        score += 0.3
    words = sentence.lower().split()
    if any(verb in words for verb in verbs):
        score += 0.4
    return min(score, 1.0)

def compute_naturalness(sentence):
    stopwords = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "and", "but", "or", "so"}
    words = sentence.lower().split()
    if not words:
        return 0.0
    stopword_count = sum(1 for w in words if w in stopwords)
    ratio = stopword_count / len(words)
    if 0.3 <= ratio <= 0.5:
        return 1.0
    else:
        return max(0.0, 1 - abs(ratio - 0.4) * 2)

def compute_style(translated, expected):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([translated, expected])
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return cosine_sim

def compute_precision(candidate_tokens, reference_tokens):
    candidate_set = set(candidate_tokens)
    reference_set = set(reference_tokens)
    if len(candidate_set) == 0:
        return 0.0
    return len(candidate_set.intersection(reference_set)) / len(candidate_set)

def compute_recall(candidate_tokens, reference_tokens):
    candidate_set = set(candidate_tokens)
    reference_set = set(reference_tokens)
    if len(reference_set) == 0:
        return 0.0
    return len(candidate_set.intersection(reference_set)) / len(reference_set)

def compute_f1(candidate_tokens, reference_tokens):
    prec = compute_precision(candidate_tokens, reference_tokens)
    rec = compute_recall(candidate_tokens, reference_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_exact_accuracy(candidate, reference):
    return 1.0 if candidate.strip() == reference.strip() else 0.0
