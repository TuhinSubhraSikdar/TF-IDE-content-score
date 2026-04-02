import os
import math
from collections import Counter
from typing import List

try:
    import docx
except ImportError:
    raise ImportError("Please install python-docx: pip install python-docx")

# -----------------------------
# 1. Load DOCX file
# -----------------------------

def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text).lower()

# -----------------------------
# 2. Tokenization (Improved)
# -----------------------------

def tokenize(text: str) -> List[str]:
    return [word.strip(".,!?;:()[]\"'") for word in text.split() if word]

# -----------------------------
# 3. N-gram Support (NEW 🔥)
# -----------------------------

def generate_ngrams(tokens: List[str], n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# -----------------------------
# 4. TF Calculation
# -----------------------------

def compute_tf(tokens: List[str]) -> dict:
    tf = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in tf.items()}

# -----------------------------
# 5. IDF Calculation (Multi-doc ready)
# -----------------------------

def compute_idf(docs_tokens: List[List[str]]) -> dict:
    N = len(docs_tokens)
    idf = {}

    all_words = set(word for doc in docs_tokens for word in doc)

    for word in all_words:
        containing_docs = sum(1 for doc in docs_tokens if word in doc)
        idf[word] = math.log((N + 1) / (containing_docs + 1)) + 1

    return idf

# -----------------------------
# 6. TF-IDF Calculation
# -----------------------------

def compute_tfidf(tf: dict, idf: dict) -> dict:
    return {word: tf[word] * idf.get(word, 0) for word in tf}

# -----------------------------
# 7. Semantic Intent Matching (Improved)
# -----------------------------

def intent_score(tfidf: dict, intent_keywords: List[str], tokens: List[str]) -> float:
    score = 0

    token_set = set(tokens)

    for keyword in intent_keywords:
        keyword = keyword.lower()

        # direct match
        if keyword in tfidf:
            score += tfidf.get(keyword, 0)

        # partial / token-level match
        parts = keyword.split()
        for p in parts:
            if p in token_set:
                score += tfidf.get(p, 0) * 0.5

    return score

# -----------------------------
# 8. LLM Optimization Score (Upgraded 🔥)
# -----------------------------

def llm_score(text: str, tokens: List[str], intent_keywords: List[str]):
    word_count = len(tokens)
    unique_words = len(set(tokens))

    keyword_coverage = sum(1 for k in intent_keywords if k.lower() in text) / len(intent_keywords)
    vocab_richness = unique_words / word_count if word_count else 0

    # Structure detection
    heading_count = text.count("\n")
    has_lists = "-" in text or "•" in text

    # Section clarity (NEW)
    sections = text.split("\n\n")
    avg_section_length = sum(len(s.split()) for s in sections) / len(sections) if sections else 0

    score = 0

    # Keyword coverage
    if keyword_coverage > 0.7:
        score += 30
    elif keyword_coverage > 0.4:
        score += 20
    else:
        score += 10

    # Vocabulary richness
    if vocab_richness > 0.5:
        score += 20

    # Structure
    if heading_count > 5:
        score += 20

    if has_lists:
        score += 10

    # Chunking (LLM critical)
    if avg_section_length < 120:
        score += 20

    return min(score, 100), {
        "keyword_coverage": keyword_coverage,
        "vocab_richness": vocab_richness,
        "avg_section_length": avg_section_length,
        "structure_score": heading_count
    }

# -----------------------------
# 9. Benchmarks
# -----------------------------

BENCHMARKS = {
    "intent": 75,
    "llm": 70
}

# -----------------------------
# 10. Final Analyzer
# -----------------------------

def analyze_document(doc_path: str, intent_keywords: List[str]):
    print("\n📄 Loading document...")
    text = load_docx(doc_path)

    tokens = tokenize(text)
    bigrams = generate_ngrams(tokens, 2)

    combined_tokens = tokens + bigrams  # 🔥 include phrases like "remote patient"

    tf = compute_tf(combined_tokens)
    docs_tokens = [combined_tokens]
    idf = compute_idf(docs_tokens)
    tfidf = compute_tfidf(tf, idf)

    intent = intent_score(tfidf, intent_keywords, tokens)

    max_possible = sum(sorted(tfidf.values(), reverse=True)[:len(intent_keywords)])
    normalized_intent = (intent / max_possible) * 100 if max_possible > 0 else 0

    llm_opt_score, diagnostics = llm_score(text, tokens, intent_keywords)

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
        print("✅ Excellent: Fully optimized for SEO + LLMs")
    elif normalized_intent >= 50 and llm_opt_score >= 50:
        print("⚠️ Moderate: Needs improvement")
    else:
        print("❌ Poor: Not optimized")

    print("\n📉 GAP ANALYSIS:")
    print(f"Intent Gap: {BENCHMARKS['intent'] - normalized_intent:.2f}%")
    print(f"LLM Gap: {BENCHMARKS['llm'] - llm_opt_score:.2f}%")

    if (BENCHMARKS['intent'] - normalized_intent) > (BENCHMARKS['llm'] - llm_opt_score):
        print("👉 Fix Priority: Add better keyword + entity coverage")
    else:
        print("👉 Fix Priority: Improve structure (headings, chunking, clarity)")

    print("\n🔥 Top Important Terms:")
    top_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, val in top_terms:
        print(f"{word}: {val:.4f}")


# -----------------------------
# 11. Run Example
# -----------------------------

if __name__ == "__main__":
    doc_path = "sample.docx"

    intent_keywords = [
        "llm processing",
        "tokenization",
        "embeddings",
        "self attention",
        "transformer model",
        "vector database",
        "rag"
    ]

    analyze_document(doc_path, intent_keywords)
