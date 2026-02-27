"""Text processing utilities for Chinese/English tokenization and stemming."""

from __future__ import annotations

import functools
import re

# Regex to find contiguous runs of CJK Unified Ideographs
_CJK_RUN_RE: re.Pattern[str] = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df"
    r"\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]+"
)

# Regex to extract English words (ASCII letters only)
_ENGLISH_WORD: re.Pattern[str] = re.compile(r"[a-zA-Z]+")

# Common English stopwords to filter out during tokenization
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "and",
        "or",
        "but",
        "not",
        "so",
        "if",
        "then",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
    }
)

# Irregular verb stems: map common inflected forms to base form
_IRREGULAR_STEMS: dict[str, str] = {
    "ran": "run",
    "running": "run",
    "runs": "run",
    "went": "go",
    "goes": "go",
    "going": "go",
    "gone": "go",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "had": "have",
    "has": "have",
    "having": "have",
    "did": "do",
    "does": "do",
    "doing": "do",
    "done": "do",
    "said": "say",
    "says": "say",
    "saying": "say",
    "made": "make",
    "makes": "make",
    "making": "make",
    "took": "take",
    "takes": "take",
    "taking": "take",
    "taken": "take",
    "came": "come",
    "comes": "come",
    "coming": "come",
    "got": "get",
    "gets": "get",
    "getting": "get",
    "gotten": "get",
    "knew": "know",
    "knows": "know",
    "knowing": "know",
    "known": "know",
    "thought": "think",
    "thinks": "think",
    "thinking": "think",
    "gave": "give",
    "gives": "give",
    "giving": "give",
    "given": "give",
    "found": "find",
    "finds": "find",
    "finding": "find",
    "told": "tell",
    "tells": "tell",
    "telling": "tell",
    "wrote": "write",
    "writes": "write",
    "writing": "write",
    "written": "write",
}


def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK ideograph (code-point range check)."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
    )


def tokenize_chinese(text: str) -> list[str]:
    """Split Chinese text into 2-grams (bigrams) using a sliding window.

    Non-CJK characters are ignored.  Returns an empty list when fewer than
    2 CJK characters are present.

    Example::

        >>> tokenize_chinese("数据库")
        ['数据', '据库']
    """
    # Use regex to extract contiguous CJK runs, then generate bigrams
    bigrams: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        run_str: str = run
        if len(run_str) < 2:
            bigrams.append(run_str)
        else:
            for i in range(len(run_str) - 1):
                bigrams.append(run_str[i] + run_str[i + 1])
    return bigrams


@functools.lru_cache(maxsize=4096)
def stem_english(word: str) -> str:
    """Simple English stemming via suffix stripping.

    Handles common inflectional suffixes: -ing, -ed, -s, -tion/-sion, -ly,
    -ment, -ness, -er, -est.  Also checks an irregular-verb lookup table.
    The input is lowercased before processing.
    """
    word = word.lower().strip()
    if not word:
        return word

    # Check irregular forms first
    if word in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[word]

    # Minimum length to attempt stripping (avoid over-stemming short words)
    if len(word) <= 3:
        return word

    # -tion -> -te (e.g. "creation" is not great, but "action" -> "act" is tricky)
    # Simplified: strip -tion, -sion
    if word.endswith("tion") and len(word) > 5:
        return word[:-4] + "te" if word[-5] == "a" else word[:-3]
    if word.endswith("sion") and len(word) > 5:
        return word[:-4] + "de" if word[-5] == "u" else word[:-3]

    # -ness (e.g. "happiness" -> "happi" -- simplified)
    if word.endswith("ness") and len(word) > 5:
        return word[:-4]

    # -ment (e.g. "development" -> "develop")
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]

    # -ly (e.g. "quickly" -> "quick")
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]

    # -ing (e.g. "running" -> "run", "making" -> "make")
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        # Doubled consonant: "running" -> "runn" -> "run"
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        # Silent-e: "making" -> "mak" -> "make"
        if len(stem) >= 2 and stem[-1] not in "aeiou" and stem[-2] in "aeiou":
            return stem + "e"
        return stem

    # -ed (e.g. "played" -> "play", "stopped" -> "stop")
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        # Doubled consonant: "stopped" -> "stopp" -> "stop"
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        # "ied" -> "y" (e.g. "studied" -> "study")
        if word.endswith("ied"):
            return word[:-3] + "y"
        return stem

    # -er (e.g. "faster" -> "fast")
    if word.endswith("er") and len(word) > 4:
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem

    # -est (e.g. "fastest" -> "fast")
    if word.endswith("est") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem

    # -ies (e.g. "flies" -> "fly", "tries" -> "try") — must check before -es
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # -es (e.g. "matches" -> "match", "boxes" -> "box")
    if word.endswith("es") and len(word) > 3:
        if word.endswith(("shes", "ches", "xes", "zes", "ses")):
            return word[:-2]
        return word[:-1]  # "goes" case is handled by irregular table

    # -s (e.g. "cats" -> "cat")
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def extract_tokens(text: str) -> list[str]:
    """Extract mixed Chinese/English tokens from text.

    - Chinese portions are converted to 2-grams via :func:`tokenize_chinese`.
    - English words are lowercased, stopwords removed, and stemmed.
    - Punctuation and numbers are discarded.

    Returns a flat list of tokens suitable for fuzzy matching.
    """
    if not text:
        return []

    tokens: list[str] = []

    # Chinese: extract CJK runs and generate 2-grams (fast regex-based)
    for run in _CJK_RUN_RE.findall(text):
        run_str: str = run
        if len(run_str) < 2:
            tokens.append(run_str)
        else:
            for i in range(len(run_str) - 1):
                tokens.append(run_str[i] + run_str[i + 1])

    # English: extract words, filter stopwords, stem
    for m in _ENGLISH_WORD.finditer(text):
        word = m.group().lower()
        if word not in _STOPWORDS:
            tokens.append(stem_english(word))

    return tokens
