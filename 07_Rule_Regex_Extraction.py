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
07_Rule_Regex_Extraction.py

MAIN PURPOSE
------------
Rule-based extraction inspired by the professor's paper, which converted
useful linguistic patterns into regular expressions for efficient matching.

REFERENCE PATTERN FAMILIES FROM THE PAPER
-----------------------------------------
- e/i alternation in contextual C/V environments
- o/u alternation
- uw/w alternation
- iy/y alternation

This template implements transparent, configurable transformation rules.
Treat them as STARTING RULES, not universal truth.
"""

import argparse
import csv
import re
from collections import Counter

TOKEN_RE=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñ]+",re.UNICODE)

def load_counts(path):
    c=Counter()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            c.update(x.lower() for x in TOKEN_RE.findall(line))
    return c

# Each rule returns alternative spellings generated from one word.
def rule_o_u(word):
    out=set()
    for i,ch in enumerate(word):
        if ch=="o": out.add(word[:i]+"u"+word[i+1:])
        elif ch=="u": out.add(word[:i]+"o"+word[i+1:])
    return out

def rule_e_i(word):
    out=set()
    for i,ch in enumerate(word):
        if ch=="e": out.add(word[:i]+"i"+word[i+1:])
        elif ch=="i": out.add(word[:i]+"e"+word[i+1:])
    return out

def rule_uw_w(word):
    out=set()
    if "uw" in word: out.add(word.replace("uw","w"))
    # Reverse expansion is intentionally restricted to existing vocabulary later.
    for m in re.finditer("w",word):
        i=m.start()
        out.add(word[:i]+"uw"+word[i+1:])
    return out

def rule_iy_y(word):
    out=set()
    if "iy" in word: out.add(word.replace("iy","y"))
    for m in re.finditer("y",word):
        i=m.start()
        out.add(word[:i]+"iy"+word[i+1:])
    return out

RULES = [
    ("e vs. i", rule_e_i),
    ("o vs. u", rule_o_u),
    ("uw vs. w", rule_uw_w),
    ("iy vs. y", rule_iy_y),
]

def extract_pairs(counts):
    vocab=set(counts)
    seen=set()
    for word in vocab:
        for label,fn in RULES:
            for alt in fn(word):
                if alt in vocab and alt!=word:
                    pair=tuple(sorted((word,alt)))
                    if pair in seen: continue
                    seen.add(pair)
                    yield pair,label

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o","--output",default="Variants.csv")
    args=ap.parse_args()

    counts=load_counts(args.input)
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["word1","word2","word1_count","word2_count","character_difference"])
        for (a,b),label in extract_pairs(counts):
            w.writerow([a,b,counts[a],counts[b],label])

if __name__=="__main__":
    main()
