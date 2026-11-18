"""Template 10: SELF-CONTAINED fallback (no imports from the other templates).

Use if the multi-file pipeline breaks or you need a quick single-file run.
  Divide & Conquer : symmetric-delete buckets -> compare only words sharing a key
  Greedy           : O(n) two-pointer check for distance 1
  DP               : bounded Levenshtein for distance 2 + traceback character_difference

Run:
  python standalone.py data.txt -o Variants.csv
  python standalone.py data.txt --max-edits 2 --min-count 2 --min-len 4
  python standalone.py data.csv --csv-column text
Output: word1, word2, word1_count, word2_count, character_difference
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+")
EMPTY = "∅"


# ---------------- load ----------------
def load_counts(path, csv_column=None, delimiter=","):
    c = Counter()
    if csv_column:
        with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                c.update(t.lower() for t in TOKEN_RE.findall(row.get(csv_column) or ""))
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                c.update(t.lower() for t in TOKEN_RE.findall(line))
    return c


# ---------------- divide & conquer candidates ----------------
def deletes(w, k):
    out, frontier = {w}, {w}
    for _ in range(k):
        frontier = {s[:i] + s[i + 1:] for s in frontier for i in range(len(s))}
        out |= frontier
    return out


def candidate_pairs(vocab, k, min_key_len=2, max_bucket=3000):
    index = defaultdict(list)
    for w in vocab:
        for key in deletes(w, k):
            if len(key) >= min_key_len:
                index[key].append(w)
    pairs, skipped = set(), 0
    for ws in index.values():
        if len(ws) > max_bucket:
            skipped += 1
        elif len(ws) > 1:
            ws.sort()
            pairs.update((ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws)))
    if skipped:
        print(f"skipped {skipped} oversized buckets (> {max_bucket})", file=sys.stderr)
    return pairs


# ---------------- greedy distance-1 check ----------------
def is_edit1(a, b):
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        i = next(k for k in range(len(a)) if a[k] != b[k])
        return a[i + 1:] == b[i + 1:]
    s, l = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(s) and s[i] == l[i]:
        i += 1
    return s[i:] == l[i + 1:]


# ---------------- DP ----------------
def bounded_lev(a, b, k):
    if abs(len(a) - len(b)) > k:
        return k + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > k:
            return k + 1
        prev = cur
    return min(prev[-1], k + 1)


def character_difference(a, b):
    """Traceback alignment; adjacent edits merged ('ñ vs. ny')."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] != b[j-1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]:
                ops.append((a[i-1], b[j-1], i - 1, i))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            ops.append((a[i-1], "", i - 1, i)); i -= 1
        else:
            ops.append(("", b[j-1], i, i)); j -= 1
    ops.reverse()
    groups = []
    for ca, cb, start, end in ops:
        if groups and start == groups[-1][3]:
            g = groups[-1]
            groups[-1] = (g[0] + ca, g[1] + cb, g[2], end)
        else:
            groups.append((ca, cb, start, end))
    return "; ".join(f"{x or EMPTY} vs. {y or EMPTY}" for x, y, _, _ in groups) or "none"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="Variants.csv")
    ap.add_argument("--max-edits", type=int, choices=[1, 2], default=1)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--csv-column")
    ap.add_argument("--delimiter", default=",")
    args = ap.parse_args()

    counts = load_counts(args.input, args.csv_column, "\t" if args.delimiter == "\\t" else args.delimiter)
    vocab = [w for w, c in counts.items() if c >= args.min_count and len(w) >= args.min_len]
    k = args.max_edits
    rows = []
    for a, b in candidate_pairs(vocab, k):
        ok = is_edit1(a, b) if k == 1 else 0 < bounded_lev(a, b, k) <= k
        if ok:
            if (counts[b], a) > (counts[a], b):     # word1 = more frequent form
                a, b = b, a
            rows.append((a, b, counts[a], counts[b], character_difference(a, b)))
    rows.sort(key=lambda r: (-(r[2] + r[3]), r[0], r[1]))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2", "word1_count", "word2_count", "character_difference"])
        w.writerows(rows)
    print(f"{len(counts)} types, {len(vocab)} used, {len(rows)} pairs -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
