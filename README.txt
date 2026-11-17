CSALGCM Practical Quiz — 10 Prepared Python Templates

These files are standalone preparation templates, not guaranteed final answers.
They are designed to be inspected and adapted by the team after seeing the released dataset.

Suggested order to study:
1) 04_EditDistance_Cdiff_Features.py
2) 05_Character_Ngram_Generalized.py
3) 08_Progressive_Filtering_Cascade.py
4) 09_Hybrid_ProfessorStyle_General.py
5) remaining specialized ingestion / precision templates

All scripts target the required Variants.csv columns:
word1, word2, word1_count, word2_count, character_difference

IMPORTANT:
- Validate all thresholds and assumptions on the released dataset.
- A close string pair is not automatically a spelling variant.
- Do not hard-code the published paper's limitations (ED=1, excluding English/proper nouns/reduplication)
  unless the quiz dataset/specification justifies them.
