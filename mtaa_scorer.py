"""
MTAA transcription scorer
==========================
Implements the decision procedure in MTAA_Manual as a tiered classifier:

  Tier 0 — normalize
  Tier 1 — exact match
  Tier 2 — codebook lookup (the ~40 word-specific accept/reject lists
           spelled out in the manual)
  Tier 3 — general rules (space errors, homophones, adjacent-key typos,
           single-letter deletion/doubling, adjacent transposition),
           gated by a "does this still sound like / read as the target"
           check and a "did this become a different real word" check
  Tier 4 — default REJECT - flagged for human review since it's the 
            least certain bucket. Accepted typos are also flagged.

The word-level rules require the response to already be ALIGNED to a
target word. Use `align_words()` to turn a (target_sentence, response_sentence)
pair into a list of (target_word_or_None, response_word_or_None) pairs,
then classify each pair with `classify_pair()`.
"""

import re # for regex-based tokenization
import json # for loading the codebook
import difflib # for sequence matching and edit distance
from pathlib import Path # for locating the codebook file

import pronouncing # for CMUdict-based pronunciation checks for homophones
import nltk # for dictionary word checks
try:
    from nltk.corpus import words as _nltk_words
    ENGLISH_WORDS = set(w.lower() for w in _nltk_words.words())
except LookupError:
    nltk.download("words")
    from nltk.corpus import words as _nltk_words
    ENGLISH_WORDS = set(w.lower() for w in _nltk_words.words())

CODEBOOK_PATH = Path(__file__).parent / "codebook.json"
with open(CODEBOOK_PATH) as f:
    CODEBOOK = json.load(f)

# ---------------------------------------------------------------------------
# Contractions: 
# ---------------------------------------------------------------------------
CONTRACTION_EXPANSIONS = {
    "'s": ["is", "has"],
    "'re": ["are"],
    "'ll": ["will"],
    "'ve": ["have"],
    "'d": ["would", "had"],
    "n't": ["not"],
}


def _strip_punct(s: str) -> str:
    return s.replace(" ", "").replace("'", "").replace("\u2019", "")


def _contraction_match(contracted: str, expanded_words):
    """'there's', ['there', 'is'] -> True"""
    if len(expanded_words) != 2:
        return False
    for suffix, options in CONTRACTION_EXPANSIONS.items():
        if contracted.endswith(suffix):
            stem = contracted[: -len(suffix)]
            if expanded_words[0] == stem and expanded_words[1] in options:
                return True
    return False


def is_contraction_equivalent(t: str, r: str) -> bool:
    """True if t and r are the same contraction, just one written as two
    words -- checked in both directions since either side could be typed
    as the contracted or the expanded form."""
    if " " in r and " " not in t:
        return _contraction_match(t, r.split(" "))
    if " " in t and " " not in r:
        return _contraction_match(r, t.split(" "))
    return False

# ---------------------------------------------------------------------------
# Homophone pairs pulled directly from the manual's "Accept homophones" list.
# Stored both directions. Can be extended as new homophones come up.
# ---------------------------------------------------------------------------
HOMOPHONE_PAIRS = [
    ("their", "there"), ("their", "they're"), ("there", "they're"),
    ("flower", "flour"),
    ("brake", "break"),
    ("hole", "whole"),
    ("pored", "poured"),
    ("ants", "aunts"),
    ("b", "bee"), ("be", "bee"),
    ("carat", "carrot"), ("caret", "carrot"),
    ("choo", "chew"), ("choo", "chu"),
    ("night", "knight"), ("nite", "knight"),
    ("wear", "where"),
    ("war", "wore"),
    ("too", "to"), ("two", "to"),
    ("stares", "stairs"),
]
HOMOPHONE_SET = set()
for a, b in HOMOPHONE_PAIRS:
    HOMOPHONE_SET.add(frozenset((a.lower(), b.lower())))

# ---------------------------------------------------------------------------
# QWERTY adjacency map for the "adjacent-key substitution" rule (3a)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# QWERTY adjacency map for the "adjacent-key substitution" rule (3a).
# Uses approximate physical key positions (real keyboards are staggered,
# each row shifted right relative to the one above) so that diagonal /
# "same column" neighbors across rows are captured too -- e.g. 'c' sits
# physically between 'd' and 'f' on the row above, not under either 's' or
# 'g'. Row offsets below (0, 0.5, 1.0) approximate standard ANSI stagger.
# ---------------------------------------------------------------------------
QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
ROW_OFFSETS = [0.0, 0.5, 1.0]
_POSITIONS = {}
for row_i, row in enumerate(QWERTY_ROWS):
    for i, ch in enumerate(row):
        _POSITIONS[ch] = (row_i, i + ROW_OFFSETS[row_i])

_ADJ = {ch: set() for ch in _POSITIONS}
_COLUMN_THRESHOLD = 0.75  # captures the two diagonal neighbors in an adjacent row
for ch_a, (row_a, x_a) in _POSITIONS.items():
    for ch_b, (row_b, x_b) in _POSITIONS.items():
        if ch_a == ch_b:
            continue
        if row_a == row_b and abs(row_a - row_b) == 0 and abs(_POSITIONS[ch_a][1] - x_b) == 1:
            _ADJ[ch_a].add(ch_b)  # same-row left/right neighbor
        elif abs(row_a - row_b) == 1 and abs(x_a - x_b) <= _COLUMN_THRESHOLD:
            _ADJ[ch_a].add(ch_b)  # diagonal/"column" neighbor in the row above/below
QWERTY_ADJACENT = _ADJ


def is_adjacent_key(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return b in QWERTY_ADJACENT.get(a, set())


# ---------------------------------------------------------------------------
# Real-word helper
# ---------------------------------------------------------------------------
def is_real_english_word(word: str) -> bool:
    """True if `word` is a dictionary English word. Deliberately dictionary-
    based rather than corpus-frequency-based: frequency counts from web text
    treat extremely common TYPOS (e.g. 'freind') as if they were words. Also checks simple inflections
    (plural -s/-es, -ed, -ing) since the base wordlist is lemma-heavy and
    otherwise misses things like 'nets' (plural of 'net')."""
    w = word.lower()
    if w in ENGLISH_WORDS:
        return True
    for suffix, strip_len in (("ies", 3), ("es", 2), ("s", 1),
                               ("ed", 2), ("ing", 3)):
        if w.endswith(suffix) and len(w) - strip_len >= 2:
            stem = w[:-strip_len]
            if stem in ENGLISH_WORDS:
                return True
            if suffix == "ies" and (stem + "y") in ENGLISH_WORDS:
                return True
    return False


def phones(word: str):
    """CMUdict phones for a word, or None if not found (i.e. not a
    standard pronounceable English word/form)."""
    p = pronouncing.phones_for_word(word.lower())
    return p[0].split() if p else None


def strip_stress(phone_list):
    return [re.sub(r"\d", "", p) for p in phone_list]


def same_pronunciation(word_a: str, word_b: str):
    """Returns True/False if both words are in CMUdict and comparable,
    otherwise None (unknown — needs human judgment or a G2P fallback)."""
    pa, pb = phones(word_a), phones(word_b)
    if pa is None or pb is None:
        return None
    return strip_stress(pa) == strip_stress(pb)


def vowel_nucleus_sequence(phone_list):
    return [p for p in strip_stress(phone_list) if p[-1:] in "012" or
            p in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                  "IH", "IY", "OW", "OY", "UH", "UW")]


# ---------------------------------------------------------------------------
# Edit operations
# ---------------------------------------------------------------------------
def edit_ops(a: str, b: str):
    """Classify a single-edit difference between two strings.
    Returns one of: 'equal', 'substitution', 'deletion', 'insertion',
    'transposition', or None if more than one edit apart."""
    sm = difflib.SequenceMatcher(None, a, b)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return "equal", None
    if len(opcodes) == 1: 
        tag, i1, i2, j1, j2 = opcodes[0]
        if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1: # single-char substitution
            return "substitution", (a[i1:i2], b[j1:j2])
        if tag == "delete" and (i2 - i1) == 1: # single-char deletion
            return "deletion", a[i1:i2]
        if tag == "insert" and (j2 - j1) == 1: # single-char insertion
            return "insertion", b[j1:j2]
    # adjacent transposition check: swap of two neighboring chars
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            if a[i] == b[j] and a[j] == b[i]:
                return "transposition", (i, j)
    return None, None


VOWELS = set("aeiou")


def is_vowel(ch: str) -> bool:
    return ch.lower() in VOWELS


def squeeze_runs(s: str) -> str:
    """Collapse consecutive repeated letters to one, e.g. 'catterpiller'
    -> 'caterpiler'. Used to detect the 'uncertain about doubling' pattern
    independent of exactly which letter got added/dropped."""
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def runs(s: str): 
    """['s','h','e','l','l'] -> [('s',1),('h',1),('e',1),('l',2)]"""
    out = []
    for ch in s:
        if out and out[-1][0] == ch:
            out[-1] = (ch, out[-1][1] + 1)
        else:
            out.append((ch, 1))
    return out


def doubling_ambiguity_ok(t: str, r: str):
    """True only when target/response differ in exactly one place, that
    difference is purely a run-length change (letter doubled vs not), the
    two share the same underlying letter skeleton, AND the letter involved
    is a consonant. Consonant doubling is orthographic noise in English
    spelling; VOWEL doubling changes the vowel's length/quality (tree->long
    'ee' vs tre->short 'e'; shell->short e vs sheel->long 'ee') which the
    manual treats as a real pronunciation change, not a typo."""
    rt, rr = runs(t), runs(r)
    if len(rt) != len(rr):
        return False
    diffs = [i for i in range(len(rt)) if rt[i] != rr[i]]
    if len(diffs) != 1:
        return False
    i = diffs[0]
    letter_t, count_t = rt[i]
    letter_r, count_r = rr[i]
    if letter_t != letter_r:
        return False  # same position doubles a DIFFERENT letter -> reject
    if count_t == count_r:
        return False
    return not is_vowel(letter_t)


# ---------------------------------------------------------------------------
# Main per-word classifier
# ---------------------------------------------------------------------------
def classify_word(target: str, response: str):
    """
    Returns a dict: {decision: 'accept'/'reject', rule: str, detail: str}.
    `detail` is set to 'flag_for_review' whenever the automated rules landed
    on the conservative default rather than a specific named rule — those
    are the cases worth a human spot-check.
    """
    t = target.strip().lower()
    r = response.strip().lower()

    if r == t:
        return {"decision": "accept", "rule": "exact_match", "detail": ""}

    if r == "":
        return {"decision": "reject", "rule": "omission", "detail": "word omitted"}

    # Tier 2: codebook (ground truth for the ~40 words the manual enumerates)
    cb = CODEBOOK.get(t)
    if cb:
        if r in cb["accept"]:
            return {"decision": "accept", "rule": "codebook", "detail": ""}
        if r in cb["reject"]:
            return {"decision": "reject", "rule": "codebook", "detail": ""}

    # Rule 1: incorrect spacing OR a dropped apostrophe (daddy's -> daddys)
    if _strip_punct(r) == _strip_punct(t):
        return {"decision": "accept", "rule": "spacing_or_apostrophe", "detail": ""}

    # Contraction written as two words (there's -> there is) or vice versa
    if is_contraction_equivalent(t, r):
        return {"decision": "accept", "rule": "contraction_equivalent", "detail": ""}

    # Rule 2: homophones
    if frozenset((t, r)) in HOMOPHONE_SET:
        return {"decision": "accept", "rule": "homophone", "detail": ""}
    if same_pronunciation(t, r) is True:
        return {"decision": "accept", "rule": "homophone_cmudict", "detail": ""}

    # Dictionary gate: if the response is itself a different real English
    # word, the manual's worked examples (bag/bat, broom/brook, web/wen,
    # our/out, many/mant) treat that as a different lexical item -> reject,
    # regardless of how small the edit distance is.
    if is_real_english_word(r):
        return {"decision": "reject", "rule": "response_is_different_real_word", "detail": ""}

    # Rule 4/6: doubling ambiguity (single-letter deletion OR repetition of
    # an already-repeated CONSONANT) -- e.g. 'buton'/'button', 'brrom'/'broom'.
    # Vowel-doubling changes (tree/tre, shell/sheel) are excluded on purpose.
    if doubling_ambiguity_ok(t, r):
        return {"decision": "accept", "rule": "doubling_ambiguity", "detail": ""}

    op, info = edit_ops(t, r)

    # Rule 5: adjacent-letter transposition, resulting in a non-word.
    # Restricted to consonant-consonant swaps: swapping a vowel's position
    # (castle/castel, porch/proch) changes which vowel sound lands where.
    if op == "transposition":
        i, j = info
        if not is_vowel(t[i]) and not is_vowel(t[j]):
            return {"decision": "accept", "rule": "adjacent_transposition", "detail": ""}
        return {"decision": "reject", "rule": "transposition_involves_vowel",
                "detail": "flag_for_review"}

    # Rule 3a: adjacent-key substitution, resulting in a non-word.
    # Same consonant-only restriction (socks/sicks, pulled/pilled swap a
    # vowel and change the vowel sound, even though the keys are adjacent).
    if op == "substitution":
        a_ch, b_ch = info
        if is_adjacent_key(a_ch, b_ch) and not is_vowel(a_ch) and not is_vowel(b_ch):
            return {"decision": "accept", "rule": "adjacent_key_substitution", "detail": ""}
        return {"decision": "reject", "rule": "non_adjacent_or_vowel_substitution",
                "detail": "flag_for_review"}

    # Single insertion/deletion of a letter that ISN'T part of a doubling
    # pattern (e.g. floating/floasting, friend/frind, fancy/facy) -- the
    # manual's own worked examples reject these, since dropping/adding a
    # distinct sound is a bigger change than doubling uncertainty.
    if op in ("insertion", "deletion"):
        return {"decision": "reject", "rule": "non_doubling_letter_change",
                "detail": "flag_for_review"}

    # Nothing matched a specific accept rule.
    edit_distance = _levenshtein(t, r)
    if edit_distance >= 3:
        return {"decision": "reject", "rule": "large_edit_distance", "detail": str(edit_distance)}

    # Ambiguous small edit not covered by a named rule -> manual's explicit
    # conservative default is REJECT; flagged for human spot-check.
    return {"decision": "reject", "rule": "default_conservative", "detail": "flag_for_review"}


def _levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Sentence-level alignment: map response words onto target words
# ---------------------------------------------------------------------------
def tokenize(sentence: str):
    return re.findall(r"[a-zA-Z']+", sentence.lower())


def align_words(target_sentence: str, response_sentence: str):
    """
    Aligns response tokens to target tokens using a word-level diff, then
    tries to repair space-related splits/joins (rule 1) before finalizing.
    Returns a list of (target_word_or_None, response_word_or_None) tuples.
    None on the target side = participant inserted an extra word.
    None on the response side = participant omitted a target word.
    """
    t_tokens = tokenize(target_sentence)
    r_tokens = tokenize(response_sentence)

    sm = difflib.SequenceMatcher(None, t_tokens, r_tokens)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((t_tokens[i1 + k], r_tokens[j1 + k]))
        elif tag == "replace":
            t_seg = t_tokens[i1:i2]
            r_seg = r_tokens[j1:j2]
            # Rule 1 repair: check if concatenating the response segment
            # (or target segment) collapses to match the other, i.e. a
            # space was inserted/omitted across a word boundary.
            if "".join(t_seg) == "".join(r_seg):
                # same letters, different spacing -> treat as one accepted unit
                pairs.append((" ".join(t_seg), " ".join(r_seg)))
            elif (len(t_seg) == 1 and _contraction_match(t_seg[0], r_seg)) or \
                 (len(r_seg) == 1 and _contraction_match(r_seg[0], t_seg)):
                # "there's" <-> "there is" -- bundle as one comparison unit
                # instead of splitting into a mismatched pair + a stray extra word
                pairs.append((" ".join(t_seg), " ".join(r_seg)))
            else:
                # fall back to one-to-one where possible, else pad with None
                for k in range(max(len(t_seg), len(r_seg))):
                    tw = t_seg[k] if k < len(t_seg) else None
                    rw = r_seg[k] if k < len(r_seg) else None
                    pairs.append((tw, rw))
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append((t_tokens[k], None))
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append((None, r_tokens[k]))
    return pairs


def score_sentence(target_sentence: str, response_sentence: str):
    """Full pipeline: align, then classify each aligned word pair."""
    pairs = align_words(target_sentence, response_sentence)
    results = []
    for tw, rw in pairs:
        if tw is None:
            results.append({"target": None, "response": rw,
                             "decision": "reject", "rule": "extra_word", "detail": ""})
        elif rw is None:
            results.append({"target": tw, "response": None,
                             "decision": "reject", "rule": "omission", "detail": ""})
        else:
            result = classify_word(tw, rw)
            result["target"] = tw
            result["response"] = rw
            results.append(result)
    overall = "accept" if all(r["decision"] == "accept" for r in results) else "reject"

    # Was every word an exact, verbatim match? If overall is "accept" but
    # some word only passed via a typo/homophone/etc. rule, that's an
    # accepted VARIATION from the target -- worth a human glance even
    # though it's not a low-confidence call.
    has_accepted_variation = overall == "accept" and any(
        r["rule"] != "exact_match" for r in results
    )

    low_confidence = any(r.get("detail") == "flag_for_review" for r in results)
    needs_review = low_confidence or has_accepted_variation

    return {
        "overall": overall,
        "needs_review": needs_review,
        "has_accepted_variation": has_accepted_variation,
        "words": results,
    }