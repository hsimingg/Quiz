"""Template 5: rule-based variant matching (the paper's regex approach) + rule mining.

A Rule says: string X may alternate with string Y, optionally only in a context
(left/right regex). A pair (a, b) matches a rule if changing ONE occurrence of X->Y
(or Y->X) in a gives b.

Rule sets - pick with pipeline.py --rules paper,hyphen,apostrophe
  paper      : the professor's patterns (e/i, o/u, uw/w, iy/y in C/V contexts)
  hyphen     : pag-ibig ~ pagibig.            Needs a tokenizer that KEEPS '-'
  apostrophe : Tagalog contractions only: iba't ~ ibat, ito'y ~ itoy, sa'yo ~ sayo.
               Needs a tokenizer that keeps "'" (keep_hyphen=True). Skips English
               (don't, we'll, actor's) and names (d'italia).
  loan       : Spanish/English loanword spellings (c/k, f/p, ñ/ny ...). OFF by default:
               context-free f/p, j/h, z/s, v/b also pair English words (pan/fan, ham/jam)
  loose      : context-free e/i, o/u, doubled vowels -> high recall, more false positives

Reduplication (araw-araw, kagalang-galang) is grammatical, not spelling, and the paper excludes it.
Pass skip_reduplicated=True to match_rule / rule_pairs to drop those pairs.

Quiz-day workflow: run pipeline.py --mode edit --mine, look at the most frequent
cdiff patterns, then add/remove rules here.   Self-test:  python rules.py
"""
import re
from collections import Counter

C = "[bcdfghjklmnñpqrstvwxyz]"   # consonant: explicit letters, so '-' and "'" never count as one
V = "[aeiou]"                    # vowel


def is_reduplicated(word):
    """Rough check for hyphenated reduplication: araw-araw, kagalang-galang, bahay-bahayan.
    True if the two halves are equal, the left ends with the right, or the right starts with the left."""
    parts = word.split("-")
    if len(parts) != 2:
        return False
    a, b = parts
    return a == b or a.endswith(b) or b.startswith(a)


class Rule:
    def __init__(self, name, x, y, left="", right="", not_right=""):
        """x <-> y, only inside the given context.
        left      : regex that must come right BEFORE the change (fixed width: Python lookbehind)
        right     : regex that must come right AFTER the change
        not_right : regex that must NOT come right after the change ('$' = word end)
        y may be '' (x is just deleted); y -> x then inserts x wherever the context fits."""
        if x == y:
            raise ValueError(f"rule {name!r}: x and y are identical")
        self.name, self.x, self.y = name, x, y
        lb = f"(?<={left})" if left else ""
        la = (f"(?={right})" if right else "") + (f"(?!{not_right})" if not_right else "")
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
    Rule("e/i before hVC", "e", "i", right="h" + V + C),        # doblehin ~ doblihin
    Rule("o/u before mC", "o", "u", right="m" + C),             # kompanya ~ kumpanya
    Rule("uw/w after C before V", "uw", "w", left=C, right=V),  # kuwento ~ kwento
    Rule("iy/y after C before V", "iy", "y", left=C, right=V),  # piyano ~ pyano
]

HYPHEN = [Rule("hyphen", "-", "")]                              # pag-ibig ~ pagibig
APOSTROPHE = [
    Rule("apostrophe 't", "'", "", left=V, right="t$"),    # iba't ~ ibat, bagama't ~ bagamat
    Rule("apostrophe 'y", "'", "", left=V, right="y$"),    # ito'y ~ itoy, siya'y ~ siyay
    Rule("apostrophe 'yo", "'", "", left=V, right="yo$"),  # sa'yo ~ sayo
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

LOAN_SAFE = [
    Rule("c/k before a,o,u", "c", "k", right="[aou]"),       # carne ~ karne
    Rule("qu/k before e,i", "qu", "k", right="[ei]"),        # queso ~ keso
    Rule("ll/ly before V", "ll", "ly", right=V),             # calle ~ calye
    Rule("ñ/ny", "ñ", "ny"),                                 # señor ~ senyor
    Rule("x/ks", "x", "ks"),                                 # taxi ~ taksi
    Rule("ch/ts", "ch", "ts"),                               # chinelas ~ tsinelas
]

LOOSE = [
    Rule("e/i (any)", "e", "i"),                             # pare ~ pari (careful!)
    Rule("o/u (any)", "o", "u"),                             # opo ~ upo  (careful!)
] + [Rule(f"{v}{v}/{v}", v + v, v) for v in "aeiou"]       # maaari ~ maari

RULE_SETS = {"paper": PAPER, "hyphen": HYPHEN, "apostrophe": APOSTROPHE, "loan": LOAN, "loan_safe": LOAN_SAFE, "loose": LOOSE}
DEFAULT_RULES = PAPER          # the rest are opt-in


def get_rules(spec="paper"):
    out = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in RULE_SETS:
            raise ValueError(f"unknown rule set {name!r}; choose from {sorted(RULE_SETS)}")
        out.extend(RULE_SETS[name])
    return out


def match_rule(a, b, rules=DEFAULT_RULES, skip_reduplicated=False):
    """Name of the first rule that turns a into b (either direction), else None.
    skip_reduplicated=True ignores pairs where either word is a hyphenated reduplication."""
    if skip_reduplicated and (is_reduplicated(a) or is_reduplicated(b)):
        return None
    for r in rules:
        if r.matches(a, b):
            return r.name
    return None


def rule_pairs(vocab, rules=DEFAULT_RULES, skip_reduplicated=False):
    """Rule-driven extraction, O(n * L): generate each word's alternatives and look them
    up in the vocabulary. Finds rule pairs at ANY edit distance (e.g. ñ/ny = 2 edits).
    skip_reduplicated=True drops pairs involving hyphenated reduplications (araw-araw)."""
    vocab = set(vocab)
    found = {}
    for w in vocab:
        if skip_reduplicated and is_reduplicated(w):
            continue
        for r in rules:
            for alt in r.alternatives(w):
                if alt != w and alt in vocab and not (skip_reduplicated and is_reduplicated(alt)):
                    found.setdefault(tuple(sorted((w, alt))), r.name)
    return found


def _gen(s):
    """Generalize a context: vowel -> V, other letter -> C, word edge -> #, '-' and "'" stay."""
    return "".join("V" if c in "aeiou" else "#" if c in "^$" else "C" if c.isalpha() else c for c in s)


def mine_patterns(pairs, top=40, width=1):
    """Tally the edit pattern of each pair as  left-context [x->y] right-context
    (contexts generalized to C/V, '#' = word edge). Use it to discover rules."""
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

##just a test for the rules, we can remove it for the actual code
if __name__ == "__main__":
    P = get_rules("paper")
    for a, b in [("kompanya", "kumpanya"), ("doblehin", "doblihin"), ("kuwento", "kwento"),
                 ("piyano", "pyano"), ("ahensiya", "ahensya"), ("lenggwahe", "lengguwahe")]:
        assert match_rule(a, b, P), (a, b)
    for a, b in [("opo", "upo"), ("pare", "pari"), ("madikit", "marikit"),
                 ("pauwi", "pawi"), ("paiyak", "payak"), ("kom-panya", "kum-panya")]:
        assert match_rule(a, b, P) is None, (a, b)          # non-variants / punctuation not a consonant
    assert match_rule("pag-ibig", "pagibig", P) is None     # punctuation rules are opt-in
    assert match_rule("pag-ibig", "pagibig", get_rules("hyphen")) == "hyphen"
    A = get_rules("apostrophe")
    for a, b, name in [("iba't", "ibat", "apostrophe 't"), ("bagama't", "bagamat", "apostrophe 't"),
                       ("ito'y", "itoy", "apostrophe 'y"), ("sa'yo", "sayo", "apostrophe 'yo")]:
        assert match_rule(a, b, A) == name, (a, b)
    for a, b in [("actor's", "actors"), ("don't", "dont"), ("we'll", "well"), ("i'm", "im"),
                 ("we're", "were"), ("d'italia", "ditalia"), ("na'ng", "nang"), ("ito'ng", "itong")]:
        assert match_rule(a, b, A) is None, (a, b)          # English, names, 'ng (nang/itong are real words)
    assert match_rule("carne", "karne", get_rules("loan")) == "c/k before a,o,u"
    assert match_rule("señor", "senyor", get_rules("loan")) == "ñ/ny"
    assert rule_pairs(["pag-ibig", "pagibig", "sayo", "sa'yo", "opo", "upo"], get_rules("hyphen,apostrophe")) == \
        {("pag-ibig", "pagibig"): "hyphen", ("sa'yo", "sayo"): "apostrophe 'yo"}
    assert is_reduplicated("araw-araw") and is_reduplicated("kagalang-galang") and is_reduplicated("bahay-bahayan")
    assert not is_reduplicated("pag-ibig") and not is_reduplicated("mag-rekord") and not is_reduplicated("kumpanya")
    H = get_rules("hyphen")
    assert match_rule("kagalang-galang", "kagalanggalang", H) == "hyphen"
    assert match_rule("kagalang-galang", "kagalanggalang", H, skip_reduplicated=True) is None
    assert match_rule("pag-ibig", "pagibig", H, skip_reduplicated=True) == "hyphen"
    assert rule_pairs(["kagalang-galang", "kagalanggalang", "pag-ibig", "pagibig"], H, skip_reduplicated=True) == \
        {("pag-ibig", "pagibig"): "hyphen"}
    try:
        get_rules("papr"); raise SystemExit("typo should have raised")
    except ValueError:
        pass
    print("all rule tests passed")
