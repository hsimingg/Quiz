"""Template 5: rule-based variant matching (the paper's regex approach) + rule mining.

A Rule says: string X may alternate with string Y, optionally only in a context
(left/right regex). A pair (a, b) matches a rule if changing ONE occurrence of X->Y
(or Y->X) in a gives b.

Three rule sets - pick with pipeline.py --rules paper,loan,loose
  PAPER : the professor's patterns (e/i, o/u, uw/w, iy/y in C/V contexts)
  LOAN  : Spanish/English loanword spellings (c/k, f/p, ñ/ny ...). OFF by default:
          context-free f/p, j/h, z/s, v/b also pair English words (pan/fan, ham/jam)
  LOOSE : context-free e/i, o/u, doubled vowels -> high recall, more false positives

Quiz-day workflow: run pipeline.py --mode edit --mine, look at the most frequent
cdiff patterns, then add/remove rules here.
"""
import re
from collections import Counter

C = "[^aeiou]"   # consonant (any non-vowel letter)
V = "[aeiou]"    # vowel


class Rule:
    def __init__(self, name, x, y, left="", right=""):
        """left must be fixed-width (Python lookbehind). Use '$' in right for word-final."""
        self.name, self.x, self.y = name, x, y
        lb = f"(?<={left})" if left else ""
        la = f"(?={right})" if right else ""
        self.fwd = re.compile(lb + re.escape(x) + la)   # x -> y
        self.bwd = re.compile(lb + re.escape(y) + la)   # y -> x

    def alternatives(self, word):
        """Every word obtainable by applying this rule at exactly one spot."""
        for rx, repl in ((self.fwd, self.y), (self.bwd, self.x)):
            for m in rx.finditer(word):
                yield word[:m.start()] + repl + word[m.end():]

    def matches(self, a, b):
        return any(alt == b for alt in self.alternatives(a))

    def __repr__(self):
        return f"Rule({self.name!r})"


PAPER = [
    Rule("e/i before hVC", "e", "i", right="h" + V + C),     # doblehin ~ doblihin
    Rule("o/u before mC", "o", "u", right="m" + C),          # kompanya ~ kumpanya
    Rule("uw/w after C before V", "uw", "w", left=C, right=V),  # kuwento ~ kwento
    Rule("iy/y after C before V", "iy", "y", left=C, right=V),  # piyano ~ pyano
]

LOAN = [
    Rule("c/k before a,o,u", "c", "k", right="[aou]"),       # carne ~ karne
    Rule("c/s before e,i", "c", "s", right="[ei]"),          # centro ~ sentro
    Rule("qu/k before e,i", "qu", "k", right="[ei]"),        # queso ~ keso
    Rule("f/p", "f", "p"),                                   # familya ~ pamilya
    Rule("v/b", "v", "b"),                                   # vaso ~ baso
    Rule("j/h", "j", "h"),                                   # jamon ~ hamon
    Rule("z/s", "z", "s"),                                   # zapatos ~ sapatos
    Rule("ll/ly before V", "ll", "ly", right=V),             # calle ~ calye
    Rule("ñ/ny", "ñ", "ny"),                                 # señor ~ senyor
    Rule("x/ks", "x", "ks"),                                 # taxi ~ taksi
    Rule("ch/ts", "ch", "ts"),                               # chinelas ~ tsinelas
]

LOOSE = [
    Rule("e/i (any)", "e", "i"),                             # pare ~ pari (careful!)
    Rule("o/u (any)", "o", "u"),                             # opo ~ upo  (careful!)
] + [Rule(f"{v}{v}/{v}", v + v, v) for v in "aeiou"]       # maaari ~ maari

RULE_SETS = {"paper": PAPER, "loan": LOAN, "loose": LOOSE}
DEFAULT_RULES = PAPER          # loan/loose are opt-in: they match English pairs like pan/fan, ham/jam


def get_rules(spec="paper"):
    out = []
    for name in spec.split(","):
        name = name.strip()
        if name:
            out.extend(RULE_SETS[name])
    return out


def match_rule(a, b, rules=DEFAULT_RULES):
    """Name of the first rule that turns a into b (either direction), else None."""
    for r in rules:
        if r.matches(a, b):
            return r.name
    return None


def rule_pairs(vocab, rules=DEFAULT_RULES):
    """Rule-driven extraction, O(n * L): generate each word's alternatives and look them
    up in the vocabulary. Finds rule pairs at ANY edit distance (e.g. ñ/ny = 2 edits)."""
    vocab = set(vocab)
    found = {}
    for w in vocab:
        for r in rules:
            for alt in r.alternatives(w):
                if alt != w and alt in vocab:
                    found.setdefault(tuple(sorted((w, alt))), r.name)
    return found


def _gen(s):
    return "".join("V" if c in "aeiou" else ("#" if c == "^" or c == "$" else "C") for c in s)


def mine_patterns(pairs, top=40, width=1):
    """Tally the edit pattern of each pair as  left-context [x->y] right-context
    (contexts generalized to C/V, '^'/'$' = word edge). Use it to discover rules."""
    from levenshtein import levenshtein
    tally, examples = Counter(), {}
    for a, b in pairs:
        _, ops = levenshtein(a, b)
        if len(ops) != 1:
            continue
        kind, ca, cb, i = ops[0]
        end = i if kind == "ins" else i + 1
        left = ("^" * width + a)[i: i + width]
        right = (a + "$" * width)[end: end + width]
        key = f"{_gen(left)}[{ca or '∅'}->{cb or '∅'}]{_gen(right)}"
        # canonical direction so e->i and i->e are counted together
        rev = f"{_gen(left)}[{cb or '∅'}->{ca or '∅'}]{_gen(right)}"
        key = min(key, rev)
        tally[key] += 1
        examples.setdefault(key, f"{a}~{b}")
    return [(k, n, examples[k]) for k, n in tally.most_common(top)]
