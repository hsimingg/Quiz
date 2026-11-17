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
06_OneTwoEdit_Linear.py

MAIN PURPOSE
------------
Fast specialized detector for candidate pairs with at most 1 or 2 edits.

WHY PREPARE THIS
----------------
The professor's published method was limited to edit distance 1, while the
class brainstorming explicitly raises the possibility of two-character
differences. This template explores that extension efficiently.

NOTE
----
For true general edit distance <= 2, DP is safest. This linear routine is a
specialized heuristic for small local insert/delete/substitute differences.
Validate it before relying on it.
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

def small_edit_count(a,b,max_edits=2):
    """Two-pointer heuristic for small local differences."""
    if abs(len(a)-len(b))>max_edits:
        return max_edits+1, []

    i=j=0
    edits=[]
    while i<len(a) and j<len(b):
        if a[i]==b[j]:
            i+=1; j+=1
            continue

        if len(a)==len(b):
            edits.append((a[i],b[j],i))
            i+=1; j+=1
        elif len(a)>len(b):
            edits.append((a[i],"∅",i))
            i+=1
        else:
            edits.append(("∅",b[j],j))
            j+=1

        if len(edits)>max_edits:
            return len(edits),edits

    while i<len(a):
        edits.append((a[i],"∅",i)); i+=1
    while j<len(b):
        edits.append(("∅",b[j],j)); j+=1

    return len(edits),edits

def diff_text(edits):
    return "; ".join(f"{x} vs. {y}" for x,y,_ in edits) if edits else "none"

def pairs(counts,max_edits=2):
    buckets=defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)
    seen=set()
    for L,words in buckets.items():
        for L2 in range(max(1,L-max_edits),L+max_edits+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b: continue
                    p=tuple(sorted((a,b)))
                    if p in seen: continue
                    seen.add(p)
                    d,edits=small_edit_count(*p,max_edits)
                    if 0<d<=max_edits:
                        yield p,edits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-edits",type=int,choices=[1,2],default=2)
    ap.add_argument("-o","--output",default="Variants.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for (a,b),edits in pairs(counts,args.max_edits):
            w.writerow([a,b,counts[a],counts[b],diff_text(edits)])

if __name__=="__main__":
    main()
