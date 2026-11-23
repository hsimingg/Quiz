"""Spelling-variant rules, based on the paper's patterns.

A Rule says: string X can also be written as Y, but only in a certain context
(what comes right before / right after). Two words are variants if applying a
rule ONCE to one word gives the other word.

    kompanya -> (o/u before m+consonant) -> kumpanya     both in the corpus -> pair

Default rules = the paper's vowel rules + hyphen (pag-ibig ~ pagibig)
              + apostrophe contractions (iba't ~ ibat)
              + loanword letters (c/k, z/s, f/p, v/b, tion/sion/syon/siyon)
              + pang-/pan- before d,l,r,s,t (pangsarili ~ pansarili)
              + a fixed list of known pairs (Maynila ~ Manila).
Left out on purpose (they pair different words): i/e and o/u everywhere (mesa/misa),
dropping h- (hari/ari), collapsing double vowels (saan/san, paa/pa), and dropping
affixes like i-, nakapag-, mai- (ibili/bili changes the meaning).
The hyphen/apostrophe rules need load_count.py to keep '-' and "'" inside words (it does by default).
"""
import re

C = "[bcdfghjklmnÃ±pqrstvwxyz]"   # any consonant
V = "[aeiou]"                    # any vowel


class Rule:
    def __init__(self, name, x, y, left="", right="", min_len=0):
        """min_len: skip words shorter than this (keeps short English words like fan/pan out)."""
        self.name, self.x, self.y, self.min_len = name, x, y, min_len
        before = f"(?<={left})" if left else ""
        after = f"(?={right})" if right else ""
        self.x_pattern = re.compile(before + re.escape(x) + after)   # finds x, to turn into y
        self.y_pattern = re.compile(before + re.escape(y) + after)   # finds y, to turn into x

    def alternatives(self, word):
        """Every word you get by applying this rule at exactly one spot."""
        if len(word) < self.min_len:
            return
        for pattern, replacement in ((self.x_pattern, self.y), (self.y_pattern, self.x)):
            for m in pattern.finditer(word):
                yield word[:m.start()] + replacement + word[m.end():]


PAPER = [
    Rule("e/i before hVC", "e", "i", right="h" + V + C),        # doblehin ~ doblihin
    Rule("o/u before mC", "o", "u", right="m" + C),             # kompanya ~ kumpanya
    Rule("o/u before lV", "o", "u", left=C, right="l" + V),     # politiko ~ pulitiko
    Rule("uw/w after C before V", "uw", "w", left=C, right=V),  # kuwento ~ kwento
    Rule("iy/y after C before V", "iy", "y", left=C, right=V),  # piyano ~ pyano
]

HYPHEN = [Rule("hyphen", "-", "")]                              # pag-ibig ~ pagibig

# Tagalog contractions only. English (don't, we'll) and names (d'italia) don't fit these contexts.
APOSTROPHE = [
    Rule("apostrophe 't", "'", "", left=V, right="t$"),         # iba't ~ ibat
    Rule("apostrophe 'y", "'", "", left=V, right="y$"),         # ito'y ~ itoy
    Rule("apostrophe 'yo", "'", "", left=V, right="yo$"),       # sa'yo ~ sayo
]

# Loanword respellings (foreign letter ~ Filipino letter). min_len=5 keeps short English
# pairs out (fan/pan, bat/vat, zip/sip); real loans are usually longer (Pilipinas, Kristo).
LOANWORD = [
    Rule("c/k before a,o,u,l,r", "c", "k", right="[aoulr]", min_len=5),  # Cristo ~ Kristo
    Rule("z/s", "z", "s", min_len=5),                                    # Luzon ~ Luson
    Rule("f/p", "f", "p", min_len=5),                                    # Filipinas ~ Pilipinas
    Rule("v/b", "v", "b", min_len=5),                                    # universidad ~ unibersidad
    Rule("tion/syon", "tion", "syon"),                                   # intervention ~ intervensyon
    Rule("tion/siyon", "tion", "siyon"),                                 # intervention ~ intervensiyon
    Rule("sion/syon", "sion", "syon"),                                   # pension ~ pensyon
    Rule("sion/siyon", "sion", "siyon"),                                 # pension ~ pensiyon
    # siyon ~ syon is already covered by "iy/y after C before V" above
]

# pang-/mang-/nang-/sing- become pan-/man-/nan-/sin- before d, l, r, s, t (pangsarili ~ pansarili).
# The pan- form is the standard one (KWF Ortograpiyang Pambansa).
PREFIX_NG = [
    Rule("ng/n before d,l,r,s,t", "ng", "n", left="pa|ma|na|si", right="[dlrst]"),
]


RULES = PAPER + HYPHEN + APOSTROPHE + LOANWORD + PREFIX_NG


def is_reduplicated(word):
    """araw-araw, kagalang-galang, bahay-bahayan: repeating a word is grammar, not spelling,
    so the paper leaves these out. True if the two halves repeat each other."""
    parts = word.split("-")
    if len(parts) != 2:
        return False
    a, b = parts
    return a == b or a.endswith(b) or b.startswith(a)


def rule_pairs(vocab, rules=RULES):
    """For every word, apply every rule; if the result is also in the vocabulary, it's a pair.
    Returns {(word1, word2): rule_name} with word1 < word2 alphabetically.
    Reduplicated words (araw-araw) are skipped."""
    vocab = set(vocab)
    pairs = {}
    for word in vocab:
        if is_reduplicated(word):
            continue
        for rule in rules:
            for alt in rule.alternatives(word):
                if alt != word and alt in vocab and not is_reduplicated(alt):
                    pairs.setdefault(tuple(sorted((word, alt))), rule.name)
    return pairs