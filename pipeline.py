"""Template 9: full pipeline (uses templates 1-8).

  load_count  -> Counter of tokens (streamed)
  candidates  -> Divide & Conquer symmetric-delete buckets (no O(n^2) all-pairs)
  edit1       -> Greedy O(n) verification when --max-edits 1
  levenshtein -> DP verification for --max-edits 2 + traceback character_difference
  rules       -> paper / loanword rules (precision), rule mining (--mine)
  classify    -> frequency labels + n-gram features
  merge_sort  -> orders the output
  union_find  -> groups pairs into lexical items

REQUIRED OUTPUT: word1, word2, word1_count, word2_count, character_difference

Typical quiz-day runs
  python pipeline.py --demo
  python pipeline.py data.txt --mode edit --mine            # explore: what differences occur?
  python pipeline.py data.txt -o Variants.csv               # rules mode (default: paper rules), edit distance 1
  python pipeline.py data.txt --rules paper,loan            # add loanword rules if the data has them
  python pipeline.py data.txt --rules paper,hyphen,apostrophe   # + pag-ibig~pagibig, iba't~ibat
  python pipeline.py data.txt --max-edits 2 --rules paper,loan,loose --extra
  python load_count.py Raw.txt  then  python pipeline.py Raw_counts.csv   # count once, rerun fast
  python pipeline.py data.txt --groups groups.csv --features features.csv
"""
import argparse
import csv
import sys
import time
from collections import Counter

from load_count import load_counts, read_counts_csv, filter_vocab, load_stopwords
from candidates import candidate_pairs
from edit1 import edit1_diff, position_of
from levenshtein import bounded_levenshtein, distance, levenshtein, character_difference
from rules import get_rules, match_rule, rule_pairs, mine_patterns
from classify import classify, ngram_similarity, pair_features
from merge_sort import merge_sort
from union_find import group_pairs

REQUIRED = ["word1", "word2", "word1_count", "word2_count", "character_difference"]
EXTRA = ["edit_distance", "rule", "label", "pct1", "pct2", "cdiff_index", "cdiff_position", "ngram_sim"]


def log(*a):
    print(*a, file=sys.stderr)


def timed(label):
    def deco(fn):
        def wrap(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            log(f"[{label}] {time.perf_counter() - t:.2f}s")
            return r
        wrap.__name__ = fn.__name__
        return wrap
    return deco


@timed("find_variants")
def find_variants(counts, vocab, max_edits=1, mode="rules", rules=None,
                  min_ngram=0.0, order="freq", min_key_len=2, max_bucket=3000, adhere_min=61.0,
                  skip_reduplicated=True):
    rules = rules if rules is not None else get_rules("paper")
    if mode == "rules":
        # rule_pairs already finds every pair a rule explains (any edit distance), so the
        # edit-distance candidates would add nothing here - skip them.
        cands = set()
        rp = rule_pairs(vocab, rules, skip_reduplicated) if rules else {}
    else:
        stats = {}
        cands = candidate_pairs(vocab, max_edits, min_key_len, max_bucket, stats)
        log(f"[candidates] {stats}")
        rp = {}
    log(f"[find_variants] vocab={len(vocab)} candidates={len(cands)} rule-generated={len(rp)}")

    rows = []
    for a, b in cands | set(rp):
        # verify: Greedy for distance 1, bounded DP otherwise
        if max_edits == 1:
            d = 1 if edit1_diff(a, b) else None
        else:
            d = bounded_levenshtein(a, b, max_edits)
            d = d if 0 < d <= max_edits else None
        rule = rp.get((a, b))
        if rule is None and d is not None and rules:
            rule = match_rule(a, b, rules, skip_reduplicated)
        if mode == "edit" and d is None:
            continue
        if mode == "rules" and rule is None:
            continue
        if min_ngram and ngram_similarity(a, b) < min_ngram:
            continue

        w1, w2 = (a, b)
        if order == "freq" and (counts[b], a) > (counts[a], b):
            w1, w2 = b, a                                   # word1 = more frequent form
        _, ops = levenshtein(w1, w2)
        idx = ops[0][3] if ops else -1
        label, p1, p2 = classify(counts[w1], counts[w2], adhere_min)
        rows.append({
            "word1": w1, "word2": w2, "word1_count": counts[w1], "word2_count": counts[w2],
            "character_difference": character_difference(w1, w2),
            "edit_distance": d if d is not None else distance(w1, w2),
            "rule": rule or "", "label": label, "pct1": p1, "pct2": p2,
            "cdiff_index": idx, "cdiff_position": position_of(idx, max(len(w1), len(w2))),
            "ngram_sim": round(ngram_similarity(w1, w2), 4),
        })
    # Divide & Conquer sort: most frequent pairs first, then alphabetical
    return merge_sort(rows, key=lambda r: (-(r["word1_count"] + r["word2_count"]), r["word1"], r["word2"]))


def write_csv(rows, path, columns, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_groups(rows, counts, path, encoding="utf-8"):
    groups = group_pairs(((r["word1"], r["word2"]) for r in rows), counts)
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerow(["group_id", "size", "dominant", "members", "counts"])
        for i, g in enumerate(groups, 1):
            w.writerow([i, len(g), g[0], " | ".join(g), " | ".join(str(counts[x]) for x in g)])
    return groups


def write_features(rows, path, encoding="utf-8"):
    feats = [pair_features(r["word1"], r["word2"], r["cdiff_index"], r["cdiff_position"],
                           r["character_difference"]) for r in rows]
    if feats:
        cols = list(dict.fromkeys(k for f in feats for k in f))
        write_csv(feats, path, cols, encoding)


def demo():
    counts = Counter({"kompanya": 51, "kumpanya": 49, "kumpleto": 86, "kompleto": 14,
                      "opo": 90, "upo": 30, "pare": 40, "pari": 40, "kuwento": 20, "kwento": 80,
                      "doblehin": 3, "doblihin": 7, "piyano": 5, "pyano": 5,
                      "senyor": 6, "señor": 4, "pamilya": 50, "familya": 9, "bata": 70, "bato": 30})
    vocab = filter_vocab(counts, min_len=3)
    rows = find_variants(counts, vocab, max_edits=1, mode="rules", rules=get_rules("paper,loan"))
    for r in rows:
        print({k: r[k] for k in REQUIRED + ["rule", "label"]})
    pairs = {frozenset((r["word1"], r["word2"])) for r in rows}
    assert frozenset(("kompanya", "kumpanya")) in pairs
    assert frozenset(("kuwento", "kwento")) in pairs
    assert frozenset(("señor", "senyor")) in pairs          # 2 edits, found by rule generation
    assert frozenset(("opo", "upo")) not in pairs           # different words, filtered
    assert frozenset(("bata", "bato")) not in pairs         # different words, filtered
    kw = next(r for r in rows if {r["word1"], r["word2"]} == {"kuwento", "kwento"})
    assert kw["word1"] == "kwento" and kw["character_difference"] == "∅ vs. u", kw
    edit_rows = find_variants(counts, vocab, max_edits=1, mode="edit")
    assert frozenset(("bata", "bato")) in {frozenset((r["word1"], r["word2"])) for r in edit_rows}
    print(group_pairs(((r["word1"], r["word2"]) for r in rows), counts))
    print("demo OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="text file (Raw.txt) or word,count CSV (Raw_counts.csv)")
    ap.add_argument("-o", "--output", default="Variants.csv")
    ap.add_argument("--demo", action="store_true")
    # input / normalization
    ap.add_argument("--split-hyphen", dest="keep_hyphen", action="store_false",
                    help="split mag-aral and ma'am into parts (default keeps internal hyphens/apostrophes)")
    ap.set_defaults(keep_hyphen=True)
    ap.add_argument("--strip-accents", action="store_true", help="fold á->a (ñ kept)")
    ap.add_argument("--min-count", type=int, default=3, help="ignore words rarer than this (3 keeps known positives)")
    ap.add_argument("--min-len", type=int, default=3, help="ignore words shorter than this")
    ap.add_argument("--stopwords", help="file with one word per line to ignore")
    # algorithm
    ap.add_argument("--min-key-len", type=int, default=2)
    ap.add_argument("--max-edits", type=int, choices=[1, 2], default=1)
    ap.add_argument("--mode", choices=["rules", "edit"], default="rules",
                    help="rules = only rule-explained pairs (precision); edit = every pair within --max-edits (recall)")
    ap.add_argument("--rules", default="paper",
                    help="comma list of: paper, hyphen, apostrophe, loan, loan_safe, loose ('' = none)")
    ap.add_argument("--keep-reduplicated", dest="skip_reduplicated", action="store_false",
                    help="also pair hyphenated reduplications (araw-araw ~ arawaraw); skipped by default")
    ap.add_argument("--min-ngram", type=float, default=0.0, help="optional 3/4-gram similarity floor (0-1)")
    ap.add_argument("--adhere-min", type=float, default=61.0, help="%% share for adhering_vs_deviating")
    ap.add_argument("--max-bucket", type=int, default=3000)
    # output
    ap.add_argument("--order", choices=["freq", "alpha"], default="freq",
                    help="freq: word1 is the more frequent form; alpha: word1 < word2")
    ap.add_argument("--extra", action="store_true", help="append " + ", ".join(EXTRA))
    ap.add_argument("--groups", help="also write grouped lexical items to this CSV")
    ap.add_argument("--features", help="also write paper features (n-grams etc.) to this CSV")
    ap.add_argument("--mine", action="store_true", help="print the most common edit patterns")
    ap.add_argument("--encoding", default="utf-8", help="use utf-8-sig only if opening the CSV in Excel")
    args = ap.parse_args()

    if args.demo:
        return demo()
    if not args.input:
        ap.error("input is required (or use --demo)")

    t0 = time.perf_counter()
    if args.input.lower().endswith(".csv"):
        # word,count file from load_count.py: already tokenized, so --split-hyphen /
        # --strip-accents do nothing here (rebuild the CSV to change them)
        counts = read_counts_csv(args.input)
    else:
        counts = load_counts(args.input, keep_hyphen=args.keep_hyphen, fold_accents=args.strip_accents)
    vocab = filter_vocab(counts, args.min_count, args.min_len, load_stopwords(args.stopwords))
    log(f"[load] tokens={sum(counts.values())} types={len(counts)} vocab_used={len(vocab)} "
        f"({time.perf_counter() - t0:.2f}s)")

    rows = find_variants(counts, vocab, args.max_edits, args.mode, get_rules(args.rules),
                         args.min_ngram, args.order, args.min_key_len, args.max_bucket, args.adhere_min,
                         args.skip_reduplicated)
    write_csv(rows, args.output, REQUIRED + (EXTRA if args.extra else []), args.encoding)
    log(f"[done] {len(rows)} pairs -> {args.output}")

    if args.groups:
        g = write_groups(rows, counts, args.groups, args.encoding)
        log(f"[done] {len(g)} groups -> {args.groups}")
    if args.features:
        write_features(rows, args.features, args.encoding)
        log(f"[done] features -> {args.features}")
    if args.mine:
        log("\nMost common single-edit patterns (C=consonant, V=vowel, #=word edge):")
        for pat, n, ex in mine_patterns((r["word1"], r["word2"]) for r in rows):
            log(f"  {n:6d}  {pat:18s} e.g. {ex}")
        log(Counter(r["rule"] or "(no rule)" for r in rows).most_common())


if __name__ == "__main__":
    main()
