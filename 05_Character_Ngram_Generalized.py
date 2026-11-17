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
05_Character_Ngram_Generalized.py

MAIN PURPOSE
------------
Professor-paper-aligned character n-gram and generalized C/V features.

FEATURES
--------
- character 3-grams
- character 4-grams
- generalized n-grams where consonants -> C and vowels -> V

This script uses n-gram overlap as a candidate score. It is a TEMPLATE:
the acceptance threshold must be validated on the released dataset.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+", re.UNICODE)
VOWELS=set("aeiouAEIOU")

def load_counts(path):
    c=Counter()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            c.update(x.lower() for x in TOKEN_RE.findall(line))
    return c

def generalized(word):
    return "".join("V" if ch in VOWELS else "C" for ch in word)

def ngrams(word,n):
    padded=f"_{word}_"
    return {padded[i:i+n] for i in range(len(padded)-n+1)}

def jaccard(a,b):
    if not a and not b: return 1.0
    u=a|b
    return len(a&b)/len(u) if u else 0.0

def score(a,b):
    values=[]
    for n in (3,4):
        values.append(jaccard(ngrams(a,n),ngrams(b,n)))
        values.append(jaccard(ngrams(generalized(a),n),ngrams(generalized(b),n)))
    return sum(values)/len(values)

def char_diff(a,b):
    out=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else "∅"
        y=b[i] if i<len(b) else "∅"
        if x!=y: out.append(f"{x} vs. {y}")
    return "; ".join(out) if out else "none"

def candidates(counts,max_length_diff=2,min_score=0.55):
    buckets=defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)

    seen=set()
    for L,words in buckets.items():
        for L2 in range(max(1,L-max_length_diff),L+max_length_diff+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b: continue
                    p=tuple(sorted((a,b)))
                    if p in seen: continue
                    seen.add(p)
                    s=score(*p)
                    if s>=min_score:
                        yield p,s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--min-score",type=float,default=0.55)
    ap.add_argument("--max-length-diff",type=int,default=2)
    ap.add_argument("-o","--output",default="Variants.csv")
    ap.add_argument("--feature-output",default="ngram_features.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    rows=list(candidates(counts,args.max_length_diff,args.min_score))

    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for (a,b),s in rows:
            w.writerow([a,b,counts[a],counts[b],char_diff(a,b)])

    with open(args.feature_output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","ngram_score","gen_word1","gen_word2"])
        for (a,b),s in rows:
            w.writerow([a,b,f"{s:.4f}",generalized(a),generalized(b)])

if __name__=="__main__":
    main()
