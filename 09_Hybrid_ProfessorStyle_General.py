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
09_Hybrid_ProfessorStyle_General.py

MAIN PURPOSE
------------
General fallback combining the professor paper's feature philosophy with
large-data candidate reduction.

PIPELINE
--------
input -> normalization -> unigram counts -> length buckets ->
cheap overlap filter -> edit-distance DP -> cdiff features ->
n-gram/generalized n-gram features -> configurable rule score -> output

IMPORTANT
---------
The final score is deliberately configurable. Use the released dataset and
manual checks to choose thresholds.
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

def gen(word):
    return "".join("V" if ch in VOWELS else "C" for ch in word)

def grams(word,n):
    p=f"_{word}_"
    return {p[i:i+n] for i in range(len(p)-n+1)}

def jacc(A,B):
    return len(A&B)/len(A|B) if A|B else 0.0

def ngram_score(a,b):
    vals=[]
    for n in (3,4):
        vals.append(jacc(grams(a,n),grams(b,n)))
        vals.append(jacc(grams(gen(a),n),grams(gen(b),n)))
    return sum(vals)/len(vals)

def lev(a,b,k=2):
    if abs(len(a)-len(b))>k:return k+1
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        if min(cur)>k:return k+1
        prev=cur
    return prev[-1]

def cdiff(a,b):
    out=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else "∅"
        y=b[i] if i<len(b) else "∅"
        if x!=y:out.append(f"{x} vs. {y}")
    return "; ".join(out) if out else "none"

def relative_frequency(count_a,count_b):
    total=count_a+count_b
    return (count_a/total,count_b/total) if total else (0.0,0.0)

def generate(counts,max_ed=2,min_ngram=0.45,min_char_overlap=0.55):
    buckets=defaultdict(list)
    for w in counts:buckets[len(w)].append(w)
    seen=set()

    for L,words in buckets.items():
        for L2 in range(max(1,L-max_ed),L+max_ed+1):
            for a in words:
                for b in buckets.get(L2,[]):
                    if a==b:continue
                    p=tuple(sorted((a,b)))
                    if p in seen:continue
                    seen.add(p)

                    char_overlap=jacc(set(p[0]),set(p[1]))
                    if char_overlap<min_char_overlap:
                        continue

                    d=lev(*p,max_ed)
                    if not (0<d<=max_ed):
                        continue

                    ns=ngram_score(*p)
                    if ns<min_ngram:
                        continue

                    fa,fb=relative_frequency(counts[p[0]],counts[p[1]])
                    yield p,{
                        "edit_distance":d,
                        "ngram_score":ns,
                        "freq_share_1":fa,
                        "freq_share_2":fb,
                        "cdiff":cdiff(*p),
                    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-edit-distance",type=int,default=2)
    ap.add_argument("--min-ngram-score",type=float,default=0.45)
    ap.add_argument("--min-char-overlap",type=float,default=0.55)
    ap.add_argument("-o","--output",default="Variants.csv")
    ap.add_argument("--feature-output",default="hybrid_features.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    rows=list(generate(counts,args.max_edit_distance,args.min_ngram_score,args.min_char_overlap))

    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for (a,b),feat in rows:
            w.writerow([a,b,counts[a],counts[b],feat["cdiff"]])

    with open(args.feature_output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","edit_distance","ngram_score",
                    "freq_share_1","freq_share_2","cdiff"])
        for (a,b),feat in rows:
            w.writerow([a,b,feat["edit_distance"],f'{feat["ngram_score"]:.4f}',
                        f'{feat["freq_share_1"]:.4f}',f'{feat["freq_share_2"]:.4f}',
                        feat["cdiff"]])

if __name__=="__main__":
    main()
