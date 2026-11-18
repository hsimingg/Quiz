"""Template 8 (Divide and Conquer): stable merge sort with key, binary search,
and divide-and-conquer word counting.

merge_sort      : orders the output rows (e.g. by rule, then by total frequency).
binary_search   : look up a word in a sorted vocabulary in O(log n).
dc_count        : count tokens by splitting the list, counting halves, merging Counters.
"""
from collections import Counter


def merge_sort(items, key=lambda x: x, reverse=False):
    items = list(items)
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort(items[:mid], key, reverse)
    right = merge_sort(items[mid:], key, reverse)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        kl, kr = key(left[i]), key(right[j])
        take_left = kl >= kr if reverse else kl <= kr      # ties -> left: stable
        if take_left:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def binary_search(sorted_items, target, key=lambda x: x):
    """Index of target in a list sorted ascending by key, or -1."""
    lo, hi = 0, len(sorted_items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        k = key(sorted_items[mid])
        if k == target:
            return mid
        if k < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def dc_count(tokens, cutoff=50_000):
    """Divide and conquer word count: split, count each half, merge the Counters."""
    if len(tokens) <= cutoff:
        return Counter(tokens)
    mid = len(tokens) // 2
    left, right = dc_count(tokens[:mid], cutoff), dc_count(tokens[mid:], cutoff)
    left.update(right)
    return left
