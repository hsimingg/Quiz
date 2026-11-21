"""Template 1: load corpus -> clean -> tokenize -> normalize -> word frequency Counter.

Input is one UTF-8 .txt file (e.g. the Tagalog Wikipedia dump Raw.txt), streamed line by line.

Run directly to write two files next to the input:
  python load_count.py Raw.txt
    -> Raw_clean.txt   the text with markup and gibberish lines removed
    -> Raw_counts.csv  word,count (most frequent first), easy to open in Excel

Tune after seeing the dataset:
  - keep_hyphen   : keep "mag-aral" as one token (True) or split it (False)
  - strip_accents : fold "pagbása" -> "pagbasa" (ñ is always kept)
  - min_len       : drop very short tokens ("ng", "sa") from the vocabulary
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

LETTERS = r"A-Za-zÀ-ÖØ-öø-ÿÑñ"
WORD_RE = re.compile(rf"[{LETTERS}]+")
WORD_HYPHEN_RE = re.compile(rf"[{LETTERS}]+(?:[-'’][{LETTERS}]+)*")

# ---------------------------------------------------------------- cleaning
# Wiki leftovers: HTML tags (<br>, <templatestyles ...>), math/code placeholders
# (formula_12, codice_3), URLs, HTML entities (&nbsp;) and stray [[ ]] {{ }}.
MARKUP_RE = re.compile(r"<[^>]*>|\b(?:formula|codice)_\d+\b|https?://\S+|&[a-z]+;|\[\[|\]\]|\{\{|\}\}")
TABLE_STARTS = ("{|", "|", "!")          # wikitable rows like  !rowspan="1" align="center"|Upuan
APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})  # curly / modifier -> straight
MIN_LETTER_SHARE = 0.6                   # below this the line is mostly symbols, IPA, formulas or another script
TAGALOG_MARKERS = frozenset("ang ng sa na mga si ay at ni nang kay ito siya noong".split())
MARKER_CHECK_WORDS = 6                   # long lines without any marker are another language


def clean_line(line):
    """Return the line with markup removed, or None if the line is junk."""
    line = " ".join(MARKUP_RE.sub(" ", line.translate(APOSTROPHES)).split())
    if not line or line.startswith(TABLE_STARTS):
        return None
    words = WORD_RE.findall(line)
    if not words:
        return None                      # "2017", "1922:", " ],"
    letters = sum(map(len, words))
    others = sum(1 for c in line if not c.isspace() and not c.isdigit()) - letters  # digits are fine: years, dates
    if letters / (letters + others) < MIN_LETTER_SHARE:
        return None
    if len(words) >= MARKER_CHECK_WORDS and not TAGALOG_MARKERS & {w.lower() for w in words}:
        return None                      # language lists, Hawaiian/Polish examples, English-only lines
    return line


def _iter_lines(path, clean=True):
    """Yield the cleaned, non-junk lines of a UTF-8 text file."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = clean_line(raw) if clean else raw.rstrip("\n")
            if line:
                yield line


def clean_file(path, out_path):
    """Write the cleaned text to out_path. Returns (lines_in, lines_kept)."""
    total = kept = 0
    with open(path, encoding="utf-8", errors="ignore") as f, \
            open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for raw in f:
            total += 1
            line = clean_line(raw)
            if line:
                kept += 1
                out.write(line + "\n")
    return total, kept


# ---------------------------------------------------------------- counting
def strip_accents(token):
    """Remove accent marks but keep ñ (a separate letter in Filipino)."""
    out = []
    for ch in token:
        if ch in "ñÑ":
            out.append(ch)
        else:
            out.append("".join(c for c in unicodedata.normalize("NFD", ch)
                               if unicodedata.category(c) != "Mn"))
    return "".join(out)


def tokenize(text, lower=True, keep_hyphen=True, fold_accents=True):
    rx = WORD_HYPHEN_RE if keep_hyphen else WORD_RE
    for tok in rx.findall(text):
        if lower:
            tok = tok.lower()
        if fold_accents:
            tok = strip_accents(tok)
        yield tok


def load_counts(path, lower=True, keep_hyphen=True, fold_accents=True, clean=True):
    """Stream the (cleaned) input and return a Counter of tokens (raw corpus is never held in RAM)."""
    counts = Counter()
    for line in _iter_lines(path, clean):
        counts.update(tokenize(line, lower, keep_hyphen, fold_accents))
    return counts

def write_counts_csv(counts, out_path, encoding="utf-8-sig"):
    """word,count sorted by count (ties alphabetical). utf-8-sig so Excel shows ñ/á correctly."""
    with open(out_path, "w", encoding=encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "count"])
        w.writerows(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

def read_counts_csv(path, encoding="utf-8-sig"):
    """Inverse of write_counts_csv: word,count -> Counter."""
    with open(path, encoding=encoding, newline="") as f:
        r = csv.reader(f)
        next(r, None)                                   # skip header
        return Counter({w: int(n) for w, n in r})

def filter_vocab(counts, min_count=1, min_len=3, stopwords=frozenset()):
    """Vocabulary used for pairing. Counts stay global (use the full Counter for output)."""
    return {w: c for w, c in counts.items()
            if c >= min_count and len(w) >= min_len and w not in stopwords}

def load_stopwords(path):
    if not path:
        return frozenset()
    with open(path, encoding="utf-8", errors="ignore") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python load_count.py Raw.txt")
    src = Path(sys.argv[1])
    clean_path = src.with_name(src.stem + "_clean.txt")
    counts_path = src.with_name(src.stem + "_counts.csv")

    total, kept = clean_file(src, clean_path)
    print(f"[clean] kept {kept:,} of {total:,} lines -> {clean_path}")
    counts = load_counts(clean_path, clean=False)
    write_counts_csv(counts, counts_path)
    print(f"[count] {sum(counts.values()):,} tokens, {len(counts):,} distinct words -> {counts_path}")
