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
10_PrecisionFirst_Validation.py

MAIN PURPOSE
------------
Precision-first detector. Require multiple pieces of evidence before accepting
a pair, reducing false positives at the cost of potentially lower recall.

EVIDENCE USED
-------------
- small edit distance
- high n-gram overlap
- high character-set overlap
- optional relative-frequency sanity check

SEMANTIC NOTE
-------------
The professor's paper also explored Word2Vec cosine similarity. This template
does NOT require an external embedding model because availability on quiz day
is unknown. A semantic check can be plugged in if a model is supplied.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+",re.UNICODE)
VOWELS=set("aeiou")

def load_counts(path):
    c=Counter()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            c.update(x.lower() for x in TOKEN_RE.findall(line))
    return c

def levenshtein(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def grams(w,n):
    p=f"_{w}_"
    return {p[i:i+n] for i in range(len(p)-n+1)}

def overlap(A,B):
    return len(A&B)/len(A|B) if A|B else 0.0

def ngram_similarity(a,b):
    return (overlap(grams(a,3),grams(b,3))+overlap(grams(a,4),grams(b,4)))/2

def diff(a,b):
    out=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else "∅"
        y=b[i] if i<len(b) else "∅"
        if x!=y:out.append(f"{x} vs. {y}")
    return "; ".join(out) if out else "none"

def accept_pair(a,b,counts,max_edit,min_ngram,min_char_overlap,min_count):
    if min(counts[a],counts[b]) < min_count:
        return False,{}
    if abs(len(a)-len(b)) > max_edit:
        return False,{}

    d=levenshtein(a,b)
    if not (0<d<=max_edit):
        return False,{"edit_distance":d}

    ng=ngram_similarity(a,b)
    if ng<min_ngram:
        return False,{"edit_distance":d,"ngram":ng}

    co=overlap(set(a),set(b))
    if co<min_char_overlap:
        return False,{"edit_distance":d,"ngram":ng,"char_overlap":co}

    return True,{"edit_distance":d,"ngram":ng,"char_overlap":co}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-edit-distance",type=int,default=1)
    ap.add_argument("--min-ngram",type=float,default=0.65)
    ap.add_argument("--min-char-overlap",type=float,default=0.70)
    ap.add_argument("--min-count",type=int,default=2)
    ap.add_argument("-o","--output",default="Variants.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    buckets=defaultdict(list)
    for w in counts:buckets[len(w)].append(w)

    seen=set()
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])

        for L,words in buckets.items():
            for L2 in range(max(1,L-args.max_edit_distance),L+args.max_edit_distance+1):
                for a in words:
                    for b in buckets.get(L2,[]):
                        if a==b:continue
                        p=tuple(sorted((a,b)))
                        if p in seen:continue
                        seen.add(p)
                        ok,_=accept_pair(*p,counts,args.max_edit_distance,
                                         args.min_ngram,args.min_char_overlap,args.min_count)
                        if ok:
                            w.writerow([p[0],p[1],counts[p[0]],counts[p[1]],diff(*p)])

if __name__=="__main__":
    main()
