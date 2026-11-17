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
03_MultiFile_Unified_Pipeline.py

MAIN PURPOSE
------------
Process many text files as one corpus and produce GLOBAL frequencies.

ALGORITHM
---------
Directory traversal -> streaming tokenization -> one Counter ->
length-bucket candidate generation -> bounded edit distance.

EXTEND
------
Add CSV/JSON readers if the released directory mixes file formats.
"""

import argparse
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+", re.UNICODE)

def iter_text_files(root):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".text", ".corpus"}:
            yield p

def build_global_counts(root):
    counts = Counter()
    for path in iter_text_files(root):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                counts.update(x.lower() for x in TOKEN_RE.findall(line))
    return counts

def levenshtein(a,b,k=1):
    if abs(len(a)-len(b)) > k:
        return k+1
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(ca!=cb)))
        if min(cur) > k:
            return k+1
        prev=cur
    return prev[-1]

def difference(a,b):
    # Basic aligned difference; replace with a richer edit script if needed.
    out=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else "∅"
        y=b[i] if i<len(b) else "∅"
        if x!=y:
            out.append(f"{x} vs. {y}")
    return "; ".join(out) if out else "none"

def pairs(counts,k=1):
    buckets=defaultdict(list)
    for w in counts:
        buckets[len(w)].append(w)
    done=set()
    for L,words in buckets.items():
        for L2 in range(max(1,L-k),L+k+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b: continue
                    p=tuple(sorted((a,b)))
                    if p in done: continue
                    done.add(p)
                    d=levenshtein(*p,k)
                    if 0<d<=k:
                        yield p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--max-edit-distance",type=int,default=1)
    ap.add_argument("-o","--output",default="Variants.csv")
    args=ap.parse_args()

    counts=build_global_counts(args.directory)
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for a,b in pairs(counts,args.max_edit_distance):
            w.writerow([a,b,counts[a],counts[b],difference(a,b)])

if __name__=="__main__":
    main()
