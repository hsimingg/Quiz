"""Template 2 (hashing searching): candidate pair generation without O(n^2).

Symmetric-delete index: every word is filed under each string obtainable by deleting up to k characters (including the word itself). Two words within edit distance k always share at least one such key, so the vocabulary is in small buckets and only words inside the same bucket are compared.

Cost ~ O(n * L^k) instead of O(n^2). Returns a SUPERSET: verify every pair with
edit1.edit1_diff (k=1) or levenshtein.bounded_levenshtein (k=2).
"""
import sys
from collections import defaultdict


def deletes(word, k):
    """All strings reachable from word by deleting 0..k characters."""
    out = {word}
    frontier = {word}
    for _ in range(k):
        nxt = set()
        for s in frontier:
            for i in range(len(s)):
                nxt.add(s[:i] + s[i + 1:])
        out |= nxt
        frontier = nxt
    return out


def candidate_pairs(vocab, max_edits=1, min_key_len=2, max_bucket=3000, stats=None):
    """vocab: iterable of words. Returns a set of sorted (a, b) pairs that MAY be within max_edits.

    min_key_len : ignore keys shorter than this (stops "ka"/"an"-style mega buckets). Side effect with k=2: two 3-letter words 2 edits apart are not paired.
                Set 0 to disable.
    max_bucket  : skip buckets larger than this (reported on stderr). Raise it if you suspect missed pairs; lower it if memory/time explode.
    stats       : for seeing the skipped words (para we can adjust the params accrdingly)

    """
    index = defaultdict(list)
    for w in set(vocab):
        for key in deletes(w, max_edits):
            if len(key) >= min_key_len:
                index[key].append(w)

    pairs, skipped, skipped_words = set(), 0, 0
    for ws in index.values():
        if len(ws) < 2:
            continue
        if max_bucket and len(ws) > max_bucket:
            skipped += 1
            skipped_words += len(ws)
            continue
        ws.sort()
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                pairs.add((ws[i], ws[j]))
    if stats is not None:
        stats.update(keys=len(index), pairs=len(pairs), skipped_count=skipped, skipped_words=skipped_words)
    if skipped:
        print(f"[candidates] skipped {skipped} oversized buckets (> {max_bucket})", file=sys.stderr)
        print(f"{skipped_words} word entries affected", file=sys.stderr)
    return pairs

""" not need for now
def length_buckets(vocab):
    Simpler divide step (what templates 01-05 used): group by length.
    Only compare buckets L and L±k. Still O(n^2) inside a bucket - use for small vocabularies.
    b = defaultdict(list)
    for w in vocab:
        b[len(w)].append(w)
    return b
"""


def brute_force_pairs(vocab, max_edits, distance_fn):
    """O(n^2) reference used only for testing on small vocabularies."""
    ws = sorted(set(vocab))
    return {(a, b) for i, a in enumerate(ws) for b in ws[i + 1:]
            if 0 < distance_fn(a, b) <= max_edits}
