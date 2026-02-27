"""Unit tests for memx.utils.text_processing — tokenization and stemming."""

from __future__ import annotations

from memx.utils.text_processing import (
    extract_tokens,
    stem_english,
    tokenize_chinese,
)

# ── Chinese 2-gram tokenization ───────────────────────────────────────


class TestTokenizeChinese:
    def test_basic_bigrams(self) -> None:
        assert tokenize_chinese("数据库") == ["数据", "据库"]

    def test_four_char_bigrams(self) -> None:
        result = tokenize_chinese("数据库管理")
        assert result == ["数据", "据库", "库管", "管理"]

    def test_single_char_returns_single(self) -> None:
        # Single CJK character returns as-is
        result = tokenize_chinese("我")
        assert result == ["我"]

    def test_empty_string(self) -> None:
        assert tokenize_chinese("") == []

    def test_no_chinese(self) -> None:
        assert tokenize_chinese("hello world") == []

    def test_mixed_text_extracts_only_cjk(self) -> None:
        # Non-CJK characters (including punctuation) are ignored
        result = tokenize_chinese("hello数据库world")
        assert result == ["数据", "据库"]

    def test_two_char_produces_one_bigram(self) -> None:
        assert tokenize_chinese("数据") == ["数据"]

    def test_punctuation_splits_runs(self) -> None:
        # Chinese punctuation splits CJK runs; bigrams are generated per-run
        result = tokenize_chinese("数据，处理")
        assert result == ["数据", "处理"]

    def test_adjacent_chars_form_bigrams(self) -> None:
        # Without punctuation, adjacent chars form bigrams across boundary
        result = tokenize_chinese("数据处理")
        assert result == ["数据", "据处", "处理"]


# ── English stemming ──────────────────────────────────────────────────


class TestStemEnglish:
    def test_running(self) -> None:
        assert stem_english("running") == "run"

    def test_runs(self) -> None:
        assert stem_english("runs") == "run"

    def test_ran(self) -> None:
        assert stem_english("ran") == "run"

    def test_played(self) -> None:
        result = stem_english("played")
        assert result == "play"

    def test_making(self) -> None:
        result = stem_english("making")
        assert result == "make"

    def test_quickly(self) -> None:
        assert stem_english("quickly") == "quick"

    def test_short_word_unchanged(self) -> None:
        assert stem_english("run") == "run"
        assert stem_english("go") == "go"

    def test_empty_string(self) -> None:
        assert stem_english("") == ""

    def test_case_insensitive(self) -> None:
        assert stem_english("Running") == "run"

    def test_cats(self) -> None:
        assert stem_english("cats") == "cat"

    def test_boxes(self) -> None:
        assert stem_english("boxes") == "box"

    def test_flies(self) -> None:
        assert stem_english("flies") == "fly"

    def test_studied(self) -> None:
        assert stem_english("studied") == "study"

    def test_faster(self) -> None:
        assert stem_english("faster") == "fast"

    def test_fastest(self) -> None:
        assert stem_english("fastest") == "fast"

    def test_development(self) -> None:
        assert stem_english("development") == "develop"

    def test_irregular_went(self) -> None:
        assert stem_english("went") == "go"

    def test_irregular_written(self) -> None:
        assert stem_english("written") == "write"

    def test_irregular_knew(self) -> None:
        assert stem_english("knew") == "know"


# ── extract_tokens (mixed Chinese/English) ────────────────────────────


class TestExtractTokens:
    def test_english_only(self) -> None:
        tokens = extract_tokens("database query optimization")
        assert "databas" in tokens or "database" in tokens  # stemmed
        assert "query" in tokens or "queri" in tokens

    def test_chinese_only(self) -> None:
        tokens = extract_tokens("数据库管理")
        assert "数据" in tokens
        assert "据库" in tokens
        assert "库管" in tokens
        assert "管理" in tokens

    def test_mixed(self) -> None:
        tokens = extract_tokens("git 数据库")
        assert "git" in tokens
        assert "数据" in tokens

    def test_stopwords_filtered(self) -> None:
        tokens = extract_tokens("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_empty_string(self) -> None:
        assert extract_tokens("") == []

    def test_whitespace_only(self) -> None:
        assert extract_tokens("   ") == []

    def test_punctuation_only(self) -> None:
        assert extract_tokens("!!!...???") == []

    def test_numbers_not_tokenized(self) -> None:
        # Pure numeric content should not generate English tokens
        tokens = extract_tokens("12345")
        assert tokens == []
