"""Find spelling-variant pairs and write Variants.csv.

Steps
  1. load_count.py  -> clean the text and count every word
  2. find pairs, one of two ways:
       --mode rules (default): rules.py applies each rule to each word; if the result is also
                               a real word, it's a pair
       --mode edit           : candidates.py + edit1.py / levenshtein.py find every pair of
                               words within 1 (or 2) edits
  3. levenshtein.py -> describe what changed between the two words ("o vs. u")
  4. write the 5 required columns

Run
  python pipeline.py Raw.txt -o Variants.csv
  python pipeline.py Raw.txt --mode edit --max-edits 1 -o Variants.csv
  python load_count.py Raw.txt   then   python pipeline.py Raw_counts.csv   (count once, rerun fast)
  python pipeline.py --demo
"""
import argparse
import csv
from collections import Counter

from load_count import load_counts, read_counts_csv, filter_vocab
from rules import RULES, rule_pairs
from candidates import candidate_pairs
from edit1 import edit1_diff
from levenshtein import bounded_levenshtein, character_difference

COLUMNS = ["word1", "word2", "word1_count", "word2_count", "character_difference"]


def edit_pairs(vocab, max_edits=1, min_key_len=2, max_bucket=3000):
    """Every pair of words within max_edits of each other.
    candidates.py narrows down which words to compare (no comparing all n^2 pairs),
    then each candidate is verified: Greedy check for 1 edit, DP for 2 edits."""
    pairs = set()
    for a, b in candidate_pairs(vocab, max_edits, min_key_len, max_bucket):
        if max_edits == 1:
            ok = edit1_diff(a, b) is not None
        else:
            ok = 0 < bounded_levenshtein(a, b, max_edits) <= max_edits
        if ok:
            pairs.add((a, b))                               # already sorted: a < b
    return pairs


def find_variants(counts, vocab, mode="rules", max_edits=1, min_key_len=2, max_bucket=3000, rules=RULES):
    """mode="rules": only pairs a rule explains (precise).
       mode="edit" : every pair within max_edits (catches more, but also pairs different words like bata/bato)."""
    if mode == "edit":
        pairs = edit_pairs(vocab, max_edits, min_key_len, max_bucket)
    else:
        pairs = set()
        for (a, b), rule in rule_pairs(vocab, rules).items():
            if rule.startswith("apostrophe"):
                # a real contraction is more common than the plain spelling (iba't 6105 vs ibat 172).
                # if the plain form wins, it's its own word (sagot, bagay), not a contraction -> skip
                with_ap, plain = (a, b) if "'" in a else (b, a)
                if counts[with_ap] < counts[plain]:
                    continue
            pairs.add((a, b))
    rows = []
    for word1, word2 in pairs:                              # word1 < word2 alphabetically
        rows.append({
            "word1": word1,
            "word2": word2,
            "word1_count": counts[word1],
            "word2_count": counts[word2],
            "character_difference": character_difference(word1, word2),
        })
    # most frequent pairs first
    rows.sort(key=lambda r: (-(r["word1_count"] + r["word2_count"]), r["word1"]))
    return rows


def write_csv(rows, path, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def demo():
    counts = Counter({"politiko": 435, "pulitiko": 1093, "kompanya": 51, "kumpanya": 49,
                      "kuwento": 20, "kwento": 80, "opo": 90, "upo": 30, "bata": 70, "bato": 30,
                      "pag-ibig": 799, "pagibig": 14, "iba't": 6105, "ibat": 172,
                      "sagot": 900, "sago't": 5, "araw-araw": 500, "arawaraw": 3})
    rows = find_variants(counts, filter_vocab(counts))
    for r in rows:
        print(r)
    pairs = {(r["word1"], r["word2"]): r for r in rows}
    assert pairs[("politiko", "pulitiko")]["character_difference"] == "o vs. u"
    assert pairs[("politiko", "pulitiko")]["word1_count"] == 435
    assert pairs[("kuwento", "kwento")]["character_difference"] == "u vs. ∅"
    assert ("kompanya", "kumpanya") in pairs
    assert ("opo", "upo") not in pairs and ("bata", "bato") not in pairs   # different words
    assert ("pag-ibig", "pagibig") in pairs and ("iba't", "ibat") in pairs
    assert ("sago't", "sagot") not in pairs                 # plain form more common -> not a contraction
    assert ("araw-araw", "arawaraw") not in pairs           # reduplication skipped

    edit_rows = find_variants(counts, filter_vocab(counts), mode="edit", max_edits=1)
    edit_found = {(r["word1"], r["word2"]) for r in edit_rows}
    assert ("politiko", "pulitiko") in edit_found and ("kuwento", "kwento") in edit_found
    assert ("bata", "bato") in edit_found                   # edit mode is looser: no rule filter
    print("demo OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="text file (Raw.txt) or word,count CSV (Raw_counts.csv)")
    ap.add_argument("-o", "--output", default="Variants.csv")
    ap.add_argument("--mode", choices=["rules", "edit"], default="rules",
                    help="rules = only rule-explained pairs (precise); edit = every pair within --max-edits")
    ap.add_argument("--max-edits", type=int, choices=[1, 2], default=1, help="edit mode only")
    ap.add_argument("--min-key-len", type=int, default=2, help="edit mode only: skip tiny lookup keys")
    ap.add_argument("--max-bucket", type=int, default=3000, help="edit mode only: skip oversized buckets")
    ap.add_argument("--min-count", type=int, default=3, help="ignore words rarer than this")
    ap.add_argument("--min-len", type=int, default=3, help="ignore words shorter than this")
    ap.add_argument("--encoding", default="utf-8", help="use utf-8-sig if opening the CSV in Excel")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return demo()
    if not args.input:
        ap.error("input is required (or use --demo)")

    if args.input.lower().endswith(".csv"):
        counts = read_counts_csv(args.input)
    else:
        counts = load_counts(args.input)
    vocab = filter_vocab(counts, args.min_count, args.min_len)

    rows = find_variants(counts, vocab, args.mode, args.max_edits, args.min_key_len, args.max_bucket)
    write_csv(rows, args.output, args.encoding)
    print(f"{len(rows)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
