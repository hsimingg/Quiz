"""Template 4 (Dynamic Programming): Levenshtein distance, bounded version, traceback,
and the character_difference string built from the ACTUAL alignment.

Why traceback matters: comparing letter-by-letter at the same index (what most of
templates 01-10 did) breaks after an insertion/deletion:
    kuwento vs kwento -> positional: "u vs. w; w vs. e; e vs. n; ..."   (wrong)
                      -> traceback : "u vs. ∅"                           (right)
"""

EMPTY = "∅"


def levenshtein(a, b):
    """Returns (distance, ops). ops = list of (op, char_in_a, char_in_b, index_in_a)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(("sub", a[i - 1], b[j - 1], i - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", a[i - 1], "", i - 1))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1], i))
            j -= 1
    return d[n][m], ops[::-1]


def distance(a, b):
    """Plain distance, two-row DP (O(min memory))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bounded_levenshtein(a, b, k):
    """Distance if <= k, else k + 1. Stops early once a whole DP row exceeds k."""
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
    return prev[-1] if prev[-1] <= k else k + 1


def group_ops(ops):
    """Merge adjacent edits into one chunk so 'ñ' vs 'ny' reads as one difference."""
    groups = []
    for kind, ca, cb, idx in ops:
        end = idx + (0 if kind == "ins" else 1)      # ins does not consume a char of a
        if groups and idx == groups[-1]["end"]:      # touches the previous chunk
            g = groups[-1]
            g["a"] += ca
            g["b"] += cb
            g["end"] = end
        else:
            groups.append({"a": ca, "b": cb, "start": idx, "end": end})
    return groups


def character_difference(a, b, sep="; ", empty=EMPTY, merge_adjacent=True):
    """'x vs. y' for every difference, read from word a to word b."""
    _, ops = levenshtein(a, b)
    if not ops:
        return "none"
    if merge_adjacent:
        parts = [(g["a"] or empty, g["b"] or empty) for g in group_ops(ops)]
    else:
        parts = [(ca or empty, cb or empty) for _, ca, cb, _ in ops]
    return sep.join(f"{x} vs. {y}" for x, y in parts)


def first_diff_index(a, b):
    _, ops = levenshtein(a, b)
    return ops[0][3] if ops else -1
