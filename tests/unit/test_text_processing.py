"""Unit tests for memx.utils.text_processing — tokenization and stemming."""

from __future__ import annotations

from memx.core.utils.text_processing import (
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


# ── STORY-031 补充测试：词干还原表覆盖、CJK bigram 边界 ───────────────


class TestStemEnglishIrregulars:
    """不规则动词词干还原表覆盖补充。"""

    def test_thought_maps_to_think(self) -> None:
        assert stem_english("thought") == "think"

    def test_given_maps_to_give(self) -> None:
        assert stem_english("given") == "give"

    def test_taken_maps_to_take(self) -> None:
        assert stem_english("taken") == "take"

    def test_gotten_maps_to_get(self) -> None:
        assert stem_english("gotten") == "get"

    def test_told_maps_to_tell(self) -> None:
        assert stem_english("told") == "tell"

    def test_found_maps_to_find(self) -> None:
        assert stem_english("found") == "find"

    def test_said_maps_to_say(self) -> None:
        assert stem_english("said") == "say"

    def test_done_maps_to_do(self) -> None:
        assert stem_english("done") == "do"

    def test_gone_maps_to_go(self) -> None:
        assert stem_english("gone") == "go"

    def test_came_maps_to_come(self) -> None:
        assert stem_english("came") == "come"


class TestStemEnglishSuffixRules:
    """后缀规则覆盖补充。"""

    def test_tion_suffix(self) -> None:
        """Words ending in -tion should be stemmed."""
        result = stem_english("creation")
        assert isinstance(result, str)
        assert len(result) < len("creation")

    def test_sion_suffix(self) -> None:
        """Words ending in -sion should be stemmed."""
        result = stem_english("decision")
        assert isinstance(result, str)
        assert len(result) < len("decision")

    def test_ness_suffix(self) -> None:
        """Words ending in -ness should be stemmed."""
        assert stem_english("happiness") == "happi"

    def test_ment_suffix_agreement(self) -> None:
        """Words ending in -ment should strip the suffix."""
        assert stem_english("agreement") == "agree"

    def test_ly_suffix(self) -> None:
        """Words ending in -ly should strip the suffix."""
        assert stem_english("slowly") == "slow"

    def test_ing_doubled_consonant(self) -> None:
        """'stopping' -> doubled 'p' -> 'stop'."""
        assert stem_english("stopping") == "stop"

    def test_ed_doubled_consonant(self) -> None:
        """'stopped' -> doubled 'p' -> 'stop'."""
        assert stem_english("stopped") == "stop"

    def test_ies_suffix(self) -> None:
        """'tries' -> 'try'."""
        assert stem_english("tries") == "try"

    def test_shes_suffix(self) -> None:
        """'pushes' -> 'push'."""
        assert stem_english("pushes") == "push"

    def test_ches_suffix(self) -> None:
        """'matches' -> 'match'."""
        assert stem_english("matches") == "match"


class TestTokenizeChineseEdgeCases:
    """CJK bigram 分词边界补充。"""

    def test_three_consecutive_runs(self) -> None:
        """Three separate CJK runs produce bigrams per run."""
        # "AB CD EF" -> 3 runs of 2 chars each -> 3 bigrams
        result = tokenize_chinese("数据 管理 系统")
        assert "数据" in result
        assert "管理" in result
        assert "系统" in result

    def test_cjk_run_of_one_char(self) -> None:
        """Single-char CJK run returns the char itself."""
        result = tokenize_chinese("a数b")
        assert result == ["数"]

    def test_long_cjk_run_bigram_count(self) -> None:
        """N chars produce N-1 bigrams."""
        text = "一二三四五六七"  # 7 chars
        result = tokenize_chinese(text)
        assert len(result) == 6  # 7-1

    def test_no_cjk_returns_empty(self) -> None:
        """Text with no CJK characters should return empty."""
        assert tokenize_chinese("hello world 123 !@#") == []


class TestExtractTokensMixed:
    """混合分词补充。"""

    def test_english_stemming_applied(self) -> None:
        """English tokens should be stemmed in extract_tokens."""
        tokens = extract_tokens("running quickly")
        assert "run" in tokens
        assert "quick" in tokens

    def test_chinese_bigrams_generated(self) -> None:
        """Chinese text should produce bigrams."""
        tokens = extract_tokens("数据库")
        assert "数据" in tokens
        assert "据库" in tokens

    def test_mixed_content_both_types(self) -> None:
        """Mixed content should produce both English stems and Chinese bigrams."""
        tokens = extract_tokens("使用git管理数据库")
        assert "git" in tokens
        assert "使用" in tokens or "数据" in tokens

    def test_stopwords_removed_from_mixed(self) -> None:
        """Stopwords should be removed from mixed content."""
        tokens = extract_tokens("the 数据库 is great")
        assert "the" not in tokens
        assert "数据" in tokens
        assert "great" in tokens or "gre" in tokens  # might be stemmed
