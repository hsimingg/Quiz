# CSALGCM Practical Quiz Template
# Prepared before the AI-use cutoff.
#
# IMPORTANT:
# - This is a template, not a claim that one rule is universally correct.
# - Inspect the released dataset and validate thresholds/rules yourselves.
# - Required output columns:
#   word1, word2, word1_count, word2_count, character_difference
# - The professor's paper emphasizes edit distance, length, cdiff, cdiff index/
#   position, 3-4 character n-grams, generalized consonant/vowel n-grams,
#   frequency evidence, and rule-based extraction.
#

"""
01_LargeFile_Streaming_DP.py

MAIN PURPOSE
------------
Memory-conscious processing for a very large plain-text corpus.

CORE STRATEGY
-------------
1. Stream the corpus line by line.
2. Normalize/tokenize incrementally.
3. Build word frequencies without retaining the raw corpus.
4. Bucket vocabulary by word length.
5. Compare only nearby length buckets.
6. Use bounded Levenshtein distance as the expensive comparison.
7. Export candidate pairs.

BEST WHEN
---------
The file is huge and loading the full corpus into RAM is risky.

MAIN TRADE-OFF
--------------
Edit distance gives orthographic closeness, not semantic identity.
The final acceptance rule must be validated on the released data.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+", re.UNICODE)

def normalize_token(token: str) -> str:
    return token.lower()

def update_counts_from_file(path: str) -> Counter:
    counts = Counter()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for raw in TOKEN_RE.findall(line):
                token = normalize_token(raw)
                if token:
                    counts[token] += 1
    return counts

def bounded_levenshtein(a: str, b: str, max_distance: int = 1) -> int:
    """Return distance if <= max_distance; otherwise return max_distance + 1."""
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if len(a) > len(b):
        a, b = b, a

    previous = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        current = [j]
        row_min = current[0]
        for i, ca in enumerate(a, start=1):
            insert_cost = current[i - 1] + 1
            delete_cost = previous[i] + 1
            replace_cost = previous[i - 1] + (ca != cb)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]

def character_difference(a: str, b: str) -> str:
    """Compact human-readable difference. Adapt if your required format changes."""
    i = j = 0
    diffs = []
    while i < len(a) or j < len(b):
        if i < len(a) and j < len(b) and a[i] == b[j]:
            i += 1; j += 1
        elif i < len(a) and j < len(b):
            diffs.append(f"{a[i]} vs. {b[j]}")
            i += 1; j += 1
        elif i < len(a):
            diffs.append(f"{a[i]} vs. ∅")
            i += 1
        else:
            diffs.append(f"∅ vs. {b[j]}")
            j += 1
    return "; ".join(diffs) if diffs else "none"

def generate_pairs(counts: Counter, max_distance: int = 1):
    buckets = defaultdict(list)
    for word in counts:
        buckets[len(word)].append(word)

    seen = set()
    for length, words in buckets.items():
        candidate_lengths = range(max(1, length - max_distance), length + max_distance + 1)
        for other_len in candidate_lengths:
            others = buckets.get(other_len, [])
            for a in words:
                for b in others:
                    if a >= b:
                        continue
                    key = (a, b)
                    if key in seen:
                        continue
                    seen.add(key)

                    d = bounded_levenshtein(a, b, max_distance)
                    if 0 < d <= max_distance:
                        yield a, b

def write_variants(path: str, counts: Counter, pairs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "word1_count", "word2_count", "character_difference"])
        for a, b in pairs:
            writer.writerow([a, b, counts[a], counts[b], character_difference(a, b)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="Variants.csv")
    parser.add_argument("--max-edit-distance", type=int, default=1)
    args = parser.parse_args()

    counts = update_counts_from_file(args.input)
    pairs = generate_pairs(counts, args.max_edit_distance)
    write_variants(args.output, counts, pairs)

if __name__ == "__main__":
    main()
