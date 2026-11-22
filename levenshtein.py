"""Levenshtein (edit distance) + the character_difference column ("o vs. u").

bounded_levenshtein : fast "is the distance <= k?" check, used by edit mode for --max-edits 2.
character_difference: aligns the two words and reads off what changed.

Comparing letter-by-letter at the same position breaks when one word is longer:
    kuwento vs kwento -> same-position: "u vs. w; w vs. e; e vs. n; ..."   (wrong)
                      -> aligned      : "u vs. ∅"                           (right)
So we use the Levenshtein table to find the real alignment, then read the edits back.
"""

EMPTY = "∅"   # shown when a letter exists in one word but not the other


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
    # walk back from the bottom-right corner to recover which edits were made
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
    """Merge edits that sit next to each other, so 'ñ' vs 'ny' reads as one difference."""
    groups = []
    for kind, ca, cb, idx in ops:
        end = idx + (0 if kind == "ins" else 1)
        if groups and idx == groups[-1]["end"]:
            g = groups[-1]
            g["a"] += ca
            g["b"] += cb
            g["end"] = end
        else:
            groups.append({"a": ca, "b": cb, "end": end})
    return groups


def character_difference(a, b):
    """'x vs. y' for every difference, read from word a to word b."""
    _, ops = levenshtein(a, b)
    if not ops:
        return "none"
    return "; ".join(f"{g['a'] or EMPTY} vs. {g['b'] or EMPTY}" for g in group_ops(ops))
