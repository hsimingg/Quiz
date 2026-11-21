"""Template 3 (Greedy two-pointer): O(n) check for edit distance EXACTLY 1, plus cdiff info.

Greedy choice: walk both strings from the left while they agree; the FIRST mismatch is
taken as the single edit, then the remainders must be identical. Never backtracks.
Correct for distance 1 (any single substitution / insertion / deletion).
"""


def position_of(index, length):
    """beginning / middle / ending, the cdiff position feature from the paper."""
    if index == 0:
        return "beginning"
    if index >= length - 1:
        return "ending"
    return "middle"


def edit1_diff(a, b):
    """None if distance != 1. Else dict: kind, a_chars, b_chars, index, position.
    a_chars/b_chars are "" for the empty side of an insertion/deletion."""
    if a == b or abs(len(a) - len(b)) > 1:
        return None
    if len(a) == len(b): # if same length, find the diff char
        i = next(k for k in range(len(a)) if a[k] != b[k])
        if a[i + 1:] != b[i + 1:]:
            return None
        kind, ca, cb, longer = "sub", a[i], b[i], a
    else: #if not same length, check the longer word and find the char that is not in the other
        s, l = (a, b) if len(a) < len(b) else (b, a)
        i = 0
        while i < len(s) and s[i] == l[i]:
            i += 1
        if s[i:] != l[i + 1:]:
            return None
        kind, longer = "indel", l
        ca, cb = ("", l[i]) if s is a else (l[i], "")
    return {"kind": kind, "a_chars": ca, "b_chars": cb, "index": i,
            "position": position_of(i, len(longer))}


def is_edit1(a, b):
    return edit1_diff(a, b) is not None
