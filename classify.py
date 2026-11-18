"""Template 6: frequency classifier + paper features (cdiff index/position, n-grams,
generalized C/V n-grams, n-gram similarity).

Frequency labels (thresholds from the paper; change as needed):
  adhering_vs_deviating : one form has >= 61% of the pair's occurrences
  competing             : both forms between 40% and 60%
"""
VOWELS = set("aeiou")


# ---------- frequency evidence (threshold rule) ----------
def classify(count_a, count_b, adhere_min=61.0):
    total = count_a + count_b
    pa = 100.0 * count_a / total if total else 0.0
    pb = 100.0 - pa if total else 0.0
    label = "adhering_vs_deviating" if max(pa, pb) >= adhere_min else "competing"
    return label, round(pa, 1), round(pb, 1)


# ---------- string features ----------
def generalize(s):
    """Consonant -> C, vowel -> V; padding/other characters unchanged."""
    return "".join("V" if c in VOWELS else "C" if c.isalpha() else c for c in s)


def ngrams_at(word, idx, n, pad="_"):
    """All n-grams of the padded word that cover character position idx."""
    p = pad * (n - 1) + word + pad * (n - 1)
    c = idx + n - 1
    return [p[s:s + n] for s in range(c - n + 1, c + 1)]


def ngram_set(word, n, pad="_"):
    p = pad + word + pad
    return {p[i:i + n] for i in range(len(p) - n + 1)}


def jaccard(A, B):
    return len(A & B) / len(A | B) if A | B else 1.0


def ngram_similarity(a, b, ns=(3, 4)):
    """Average Jaccard overlap of character 3- and 4-grams (surface only; the
    generalized C/V version is NOT averaged in because it inflates unrelated pairs)."""
    return sum(jaccard(ngram_set(a, n), ngram_set(b, n)) for n in ns) / len(ns)


def pair_features(a, b, diff_index, diff_position, cdiff, ns=(3, 4)):
    f = {"word1": a, "word2": b, "len1": len(a), "len2": len(b),
         "cdiff": cdiff, "cdiff_index": diff_index, "cdiff_position": diff_position,
         "ngram_sim": round(ngram_similarity(a, b), 4)}
    for n in ns:
        for w, tag in ((a, "w1"), (b, "w2")):
            idx = min(max(diff_index, 0), max(len(w) - 1, 0))
            for k, g in enumerate(ngrams_at(w, idx, n), 1):
                f[f"{n}gram{k}_{tag}"] = g
                f[f"{n}gram{k}gen_{tag}"] = generalize(g)
    return f
