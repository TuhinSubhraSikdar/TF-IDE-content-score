import math
import re
from collections import Counter
from typing import List

try:
    import docx
except ImportError:
    raise ImportError("Please install python-docx: pip install python-docx")

# -----------------------------
# CONFIG 🔥
# -----------------------------
DEBUG_MODE = True   # Toggle ON/OFF for testing
STRICT_MODE = True  # Harder scoring

# -----------------------------
# STOPWORDS
# -----------------------------
STOPWORDS = set([
    "the", "and", "to", "of", "in", "a", "is", "for", "on", "with",
    "that", "by", "as", "at", "from", "it", "an", "be", "this",
    "you", "your"
])

# -----------------------------
# LOAD DOCX
# -----------------------------
def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs]).lower()

# -----------------------------
# TOKENIZE
# -----------------------------
def tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    return [w for w in words if w not in STOPWORDS]

# -----------------------------
# NGRAMS
# -----------------------------
def generate_ngrams(tokens: List[str], n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# -----------------------------
# TF
# -----------------------------
def compute_tf(tokens: List[str]) -> dict:
    tf = Counter(tokens)
    total = len(tokens)
    return {k: v / total for k, v in tf.items()}

# -----------------------------
# IDF
# -----------------------------
def compute_idf(docs_tokens: List[List[str]]) -> dict:
    N = len(docs_tokens)
    idf = {}

    all_words = set(word for doc in docs_tokens for word in doc)

    for word in all_words:
        containing_docs = sum(1 for doc in docs_tokens if word in doc)
        idf[word] = math.log((1 + N) / (1 + containing_docs)) + 1

    return idf

# -----------------------------
# TF-IDF
# -----------------------------
def compute_tfidf(tf: dict, idf: dict) -> dict:
    return {k: tf[k] * idf.get(k, 0) for k in tf}

# -----------------------------
# INTENT SCORE
# -----------------------------
def intent_score(tfidf: dict, intent_keywords: List[str]) -> float:
    score = 0

    for keyword in intent_keywords:
        keyword = keyword.lower()

        if keyword in tfidf:
            score += tfidf[keyword] * 2
        else:
            parts = keyword.split()
            part_score = sum(tfidf.get(p, 0) for p in parts)
            score += part_score * 0.5

    return score / len(intent_keywords)

# -----------------------------
# LLM SCORE
# -----------------------------
def llm_score(text: str, tokens: List[str], intent_keywords: List[str]):
    word_count = len(tokens)
    unique_words = len(set(tokens))

    keyword_hits = sum(1 for k in intent_keywords if k in text)
    keyword_coverage = keyword_hits / len(intent_keywords)

    vocab_richness = unique_words / word_count if word_count else 0

    sections = text.split("\n\n")
    avg_section_length = sum(len(s.split()) for s in sections) / len(sections) if sections else 0

    heading_count = len([line for line in text.split("\n") if len(line.split()) < 10])

    score = 0

    # Keyword coverage
    if keyword_coverage > 0.6:
        score += 30
    elif keyword_coverage > 0.3:
        score += 20
    else:
        score += 10

    # Vocabulary
    if vocab_richness > 0.5:
        score += 20

    # Structure
    if heading_count > 5:
        score += 20

    # Chunking
    if 40 <= avg_section_length <= 150:
        score += 30
    elif avg_section_length < 40:
        score += 10

    # STRICT MODE PENALTIES 🔥
    if STRICT_MODE:
        if keyword_coverage < 0.3:
            score -= 15
        if avg_section_length < 40:
            score -= 20
        if not any(x in text for x in ["buy", "sale", "supplier"]):
            score -= 15

    return max(min(score, 100), 0), {
        "keyword_coverage": keyword_coverage,
        "vocab_richness": vocab_richness,
        "avg_section_length": avg_section_length,
        "structure_score": heading_count
    }

# -----------------------------
# BENCHMARKS (STRICT)
# -----------------------------
BENCHMARKS = {
    "intent": 90 if STRICT_MODE else 75,
    "llm": 85 if STRICT_MODE else 70
}

# -----------------------------
# ANALYZER
# -----------------------------
def analyze_document(doc_path: str, intent_keywords: List[str]):
    print("\n📄 Loading document...")
    text = load_docx(doc_path)

    tokens = tokenize(text)
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)

    combined_tokens = tokens + bigrams + trigrams

    tf = compute_tf(combined_tokens)
    idf = compute_idf([combined_tokens])
    tfidf = compute_tfidf(tf, idf)

    intent = intent_score(tfidf, intent_keywords)

    max_possible = max(tfidf.values()) if tfidf else 1
    normalized_intent = (intent / max_possible) * 100

    llm_opt_score, diagnostics = llm_score(text, tokens, intent_keywords)

    # DEBUG MODE (SIMULATE BAD SCORES)
    if DEBUG_MODE:
        normalized_intent *= 0.5
        llm_opt_score *= 0.6

    print("\n📊 RESULTS:")
    print(f"Target Intent Score: {BENCHMARKS['intent']}%")
    print(f"Target LLM Score: {BENCHMARKS['llm']}%")

    print(f"Intent Match Score: {normalized_intent:.2f}% (Gap: {BENCHMARKS['intent'] - normalized_intent:.2f}%)")
    print(f"LLM Optimization Score: {llm_opt_score:.2f}% (Gap: {BENCHMARKS['llm'] - llm_opt_score:.2f}%)")

    print("\n🔍 Diagnostics:")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    print("\n🧠 FINAL CONCLUSION:")

    if normalized_intent >= BENCHMARKS['intent'] and llm_opt_score >= BENCHMARKS['llm']:
        print("✅ Excellent: Fully optimized")
    elif normalized_intent >= 50 and llm_opt_score >= 50:
        print("⚠️ Moderate: Needs improvement")
    else:
        print("❌ Poor: Not optimized")

    print("\n📉 GAP ANALYSIS:")
    print(f"Intent Gap: {BENCHMARKS['intent'] - normalized_intent:.2f}%")
    print(f"LLM Gap: {BENCHMARKS['llm'] - llm_opt_score:.2f}%")

    if (BENCHMARKS['intent'] - normalized_intent) > (BENCHMARKS['llm'] - llm_opt_score):
        print("👉 Fix Priority: Improve keyword coverage")
    else:
        print("👉 Fix Priority: Improve structure")

    print("\n🔥 Top Important Terms:")
    top_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, val in top_terms:
        print(f"{word}: {val:.4f}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    doc_path = "sample.docx"

    intent_keywords = [
        "wooden access mats",
        "wood mats for sale",
        "access mats supplier",
        "buy wooden access mats",
        "ground protection mats",
        "crane mats"
    ]

    analyze_document(doc_path, intent_keywords)