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
02_CSV_Bucketed_DP.py

MAIN PURPOSE
------------
Structured CSV/TSV input with configurable text column.

ALGORITHM
---------
CSV parsing -> tokenization -> frequency counting -> length bucketing ->
bounded edit-distance DP -> required output.

EDIT THESE FIRST
----------------
--text-column, --delimiter, and --max-edit-distance
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+", re.UNICODE)

def tokens(text):
    for t in TOKEN_RE.findall(text or ""):
        yield t.lower()

def load_csv_counts(path, text_column, delimiter=","):
    counts = Counter()
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"Column {text_column!r} not found. Found: {reader.fieldnames}")
        for row in reader:
            counts.update(tokens(row.get(text_column, "")))
    return counts

def bounded_levenshtein(a, b, k=1):
    if abs(len(a) - len(b)) > k:
        return k + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))
            row_min = min(row_min, current[-1])
        if row_min > k:
            return k + 1
        previous = current
    return previous[-1]

def cdiff(a, b):
    diffs = []
    m = max(len(a), len(b))
    for i in range(m):
        x = a[i] if i < len(a) else "∅"
        y = b[i] if i < len(b) else "∅"
        if x != y:
            diffs.append(f"{x} vs. {y}")
    return "; ".join(diffs) if diffs else "none"

def candidate_pairs(counts, k=1):
    buckets = defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)

    emitted = set()
    for L, words in buckets.items():
        for L2 in range(max(1, L-k), L+k+1):
            for a in words:
                for b in buckets.get(L2, []):
                    if a == b:
                        continue
                    pair = tuple(sorted((a, b)))
                    if pair in emitted:
                        continue
                    emitted.add(pair)
                    d = bounded_levenshtein(pair[0], pair[1], k)
                    if 0 < d <= k:
                        yield pair

def write_output(path, counts, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for a,b in pairs:
            w.writerow([a,b,counts[a],counts[b],cdiff(a,b)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--text-column", required=True)
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--max-edit-distance", type=int, default=1)
    ap.add_argument("-o","--output", default="Variants.csv")
    args = ap.parse_args()

    counts = load_csv_counts(args.input, args.text_column, args.delimiter)
    write_output(args.output, counts, candidate_pairs(counts, args.max_edit_distance))

if __name__ == "__main__":
    main()
