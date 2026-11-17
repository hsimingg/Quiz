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
04_EditDistance_Cdiff_Features.py

MAIN PURPOSE
------------
Professor-paper-aligned feature extraction:
- edit distance
- string lengths
- character difference (cdiff)
- cdiff index
- cdiff position

This template intentionally exposes FEATURES first. Classification rules should
be tuned only after examining the released dataset.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+", re.UNICODE)

def load_counts(path):
    c=Counter()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            c.update(x.lower() for x in TOKEN_RE.findall(line))
    return c

def edit_distance_and_script(a,b):
    m,n=len(a),len(b)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j]=min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+(a[i-1]!=b[j-1])
            )

    # Backtrack one minimum edit script.
    i,j=m,n
    edits=[]
    while i>0 or j>0:
        if i>0 and j>0 and a[i-1]==b[j-1] and dp[i][j]==dp[i-1][j-1]:
            i-=1; j-=1
        elif i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+1:
            edits.append(("replace", i-1, a[i-1], b[j-1]))
            i-=1; j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1:
            edits.append(("delete", i-1, a[i-1], "∅"))
            i-=1
        else:
            edits.append(("insert", i, "∅", b[j-1]))
            j-=1
    edits.reverse()
    return dp[m][n], edits

def cdiff_features(a,b):
    d, edits = edit_distance_and_script(a,b)
    if not edits:
        return {
            "edit_distance":0, "cdiff":"none", "cdiff_index":-1,
            "cdiff_position":"none"
        }

    parts=[f"{old} vs. {new}" for _,_,old,new in edits]
    first_idx=edits[0][1]
    denom=max(1,len(a)-1)
    ratio=first_idx/denom
    if ratio <= 0.25:
        pos="beginning"
    elif ratio >= 0.75:
        pos="ending"
    else:
        pos="middle"

    return {
        "edit_distance":d,
        "cdiff":"; ".join(parts),
        "cdiff_index":first_idx,
        "cdiff_position":pos
    }

def candidates(counts, max_distance=1):
    buckets=defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)
    seen=set()
    for L,words in buckets.items():
        for L2 in range(max(1,L-max_distance),L+max_distance+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b: continue
                    p=tuple(sorted((a,b)))
                    if p in seen: continue
                    seen.add(p)
                    feat=cdiff_features(*p)
                    if 0 < feat["edit_distance"] <= max_distance:
                        yield p, feat

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-edit-distance",type=int,default=1)
    ap.add_argument("-o","--output",default="Variants.csv")
    ap.add_argument("--feature-output",default="features_debug.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    rows=list(candidates(counts,args.max_edit_distance))

    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for (a,b),feat in rows:
            w.writerow([a,b,counts[a],counts[b],feat["cdiff"]])

    with open(args.feature_output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","edit_distance","length1","length2",
                    "cdiff","cdiff_index","cdiff_position"])
        for (a,b),feat in rows:
            w.writerow([a,b,feat["edit_distance"],len(a),len(b),
                        feat["cdiff"],feat["cdiff_index"],feat["cdiff_position"]])

if __name__=="__main__":
    main()
