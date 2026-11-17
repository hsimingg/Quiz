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
08_Progressive_Filtering_Cascade.py

MAIN PURPOSE
------------
Efficiency-first candidate pruning:
cheap filters first -> expensive DP last.

DEFAULT FILTERS
---------------
1. length difference
2. first/last-character compatibility (configurable)
3. character-set overlap
4. bounded edit distance

The exact filter order and thresholds should be measured on the released data.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+",re.UNICODE)

def load_counts(path):
    c=Counter()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            c.update(x.lower() for x in TOKEN_RE.findall(line))
    return c

def char_jaccard(a,b):
    A=set(a); B=set(b)
    return len(A&B)/len(A|B) if A|B else 0.0

def bounded_lev(a,b,k):
    if abs(len(a)-len(b))>k: return k+1
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        if min(cur)>k: return k+1
        prev=cur
    return prev[-1]

def diff(a,b):
    out=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else "∅"
        y=b[i] if i<len(b) else "∅"
        if x!=y: out.append(f"{x} vs. {y}")
    return "; ".join(out) if out else "none"

def passes_cheap_filters(a,b,max_len_diff,min_char_overlap,require_edge_match=False):
    if abs(len(a)-len(b))>max_len_diff:
        return False
    if require_edge_match and not (a[0]==b[0] or a[-1]==b[-1]):
        return False
    if char_jaccard(a,b)<min_char_overlap:
        return False
    return True

def generate(counts,max_len_diff=2,min_char_overlap=0.6,max_edit_distance=2,require_edge_match=False):
    buckets=defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)

    seen=set()
    for L,words in buckets.items():
        for L2 in range(max(1,L-max_len_diff),L+max_len_diff+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b: continue
                    p=tuple(sorted((a,b)))
                    if p in seen: continue
                    seen.add(p)

                    if not passes_cheap_filters(*p,max_len_diff,min_char_overlap,require_edge_match):
                        continue
                    d=bounded_lev(*p,max_edit_distance)
                    if 0<d<=max_edit_distance:
                        yield p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-length-diff",type=int,default=2)
    ap.add_argument("--min-char-overlap",type=float,default=0.6)
    ap.add_argument("--max-edit-distance",type=int,default=2)
    ap.add_argument("--require-edge-match",action="store_true")
    ap.add_argument("-o","--output",default="Variants.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    pairs=generate(counts,args.max_length_diff,args.min_char_overlap,
                   args.max_edit_distance,args.require_edge_match)

    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for a,b in pairs:
            w.writerow([a,b,counts[a],counts[b],diff(a,b)])

if __name__=="__main__":
    main()
