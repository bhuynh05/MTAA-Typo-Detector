# MTAA Transcription Scorer

Scores participant transcriptions against target sentences using the
decision rules in the `MTAA Manual`, plus the ~40 word-specific
accept/reject lists the manual already spells out.

## Files
- `codebook.json` — the manual's per-word accept/reject lists, parsed
  directly. This is ground truth: any response matching one of these is decided
  immediately, no judgment call involved.
- `mtaa_scorer.py` — the classifier. `classify_word(target, response)`
  scores one word pair; `score_sentence(target_sentence, response_sentence)`
  aligns two full sentences word-by-word and scores every pair.
- `score_csv.py` — CLI that runs the scorer over a CSV.
- `validate.py` — regression test against ~40 worked examples pulled
  straight from the manual's own text (not the codebook — the separate
  illustrative examples in sections 2–8). 40/42 currently pass.

## How it decides (in order)
1. **Exact match** → accept.
2. **Codebook lookup** → accept/reject per the manual's explicit lists.
3. **Spacing** → accept if removing all spaces makes target and
   response identical.
4. **Homophones** → accept, via an explicit list from the manual
   plus a CMU Pronouncing Dictionary phoneme check.
5. **Real-word check** → if the response is itself a different, real
   English word, reject as a different lexical item (e.g. bag→bat,
   broom→brook, web→wen).
6. **Doubling ambiguity** → accept single-letter deletion/
   repetition of an already-doubled *consonant* (buton/button,
   catterpiller/caterpillar). Vowel doubling is excluded on purpose —
   tree→tre and shell→sheel change the vowel's length/quality, which is treated
   as a real pronunciation change.
8. **Adjacent-letter transposition** and **adjacent-key
   substitution** → accept only when the letters involved are
   both consonants, for the same reason as above.
9. **Default** → reject (when there is reasonable ambiguity, code the response as incorrect).

Every word that lands on a rule with low confidence (a substitution/
insertion/deletion that isn't the doubling case, or a transposition/
substitution involving a vowel) is tagged `flag_for_review` in the
output, and the sentence gets `needs_review = True`. 

## Limitations
- Vowel-involving transpositions are the single hardest case (e.g. "freind"
  for "friend" was accepted via letter transposition, but the same swap pattern is
  rejected elsewhere — "castle"→"castel" — because it changes pronunciation).
  The scorer defaults to reject+flag here.
- Morphological changes (verb tense, plurals) and whole-syllable omission
  aren't separately modeled — they mostly fall through to the default (reject + flag).
- Word-level alignment between target and response sentences uses a
  standard sequence diff. It handles single-word substitutions and the
  space-repair case well; it's not built to handle a participant
  reordering multiple words or dropping several words in a row — those
  will show up as `extra_word`/`omission` entries you'll want to eyeball.

## Setup
```
pip install pronouncing nltk
python -c "import nltk; nltk.download('words')"
```

## Running it
```
python score_csv.py YOUR_DATA.csv scored_output.csv \
    --target-col TARGET_SENTENCE --response-col TRANSCRIBED_SENTENCE
```
Output columns added: `overall_decision`, `needs_review`, `word_details`
(a JSON breakdown per word, for auditing).
# MTAA-Typo-Detector
