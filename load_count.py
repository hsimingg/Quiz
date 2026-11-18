"""Template 1: load corpus -> tokenize -> normalize -> word frequency Counter.

Handles: a single .txt file (streamed line by line), a CSV/TSV file (pick a text
column), or a directory of text files (one global Counter).

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


def tokenize(text, lower=True, keep_hyphen=False, fold_accents=False):
    rx = WORD_HYPHEN_RE if keep_hyphen else WORD_RE
    for tok in rx.findall(text):
        if lower:
            tok = tok.lower()
        if fold_accents:
            tok = strip_accents(tok)
        yield tok


def _iter_lines(path, csv_column=None, delimiter=","):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".txt", ".text", ".corpus", ".md"}:
                with p.open(encoding="utf-8", errors="ignore") as f:
                    yield from f
        return
    if csv_column is not None:
        with path.open(encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if csv_column not in (reader.fieldnames or []):
                sys.exit(f"Column {csv_column!r} not found. Columns: {reader.fieldnames}")
            for row in reader:
                yield row.get(csv_column) or ""
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
        yield from f


def load_counts(path, lower=True, keep_hyphen=False, fold_accents=False,
                csv_column=None, delimiter=","):
    """Stream the input and return a Counter of tokens (raw corpus is never held in RAM)."""
    counts = Counter()
    for line in _iter_lines(path, csv_column, delimiter):
        counts.update(tokenize(line, lower, keep_hyphen, fold_accents))
    return counts


def filter_vocab(counts, min_count=1, min_len=3, stopwords=frozenset()):
    """Vocabulary used for pairing. Counts stay global (use the full Counter for output)."""
    return {w: c for w, c in counts.items()
            if c >= min_count and len(w) >= min_len and w not in stopwords}


def load_stopwords(path):
    if not path:
        return frozenset()
    with open(path, encoding="utf-8", errors="ignore") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


# backwards-compatible names from the first draft
def load_words(path, lower=True, encoding="utf-8"):
    with open(path, encoding=encoding, errors="ignore") as f:
        return [t for line in f for t in tokenize(line, lower)]


def count_words(words, min_len=3, stopwords=frozenset()):
    return Counter(w for w in words if len(w) >= min_len and w not in stopwords)
