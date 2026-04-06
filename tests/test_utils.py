"""Tests for pure utility functions in utils/.

Impure functions NOT tested here:
- clean.standardize_social_ats: uses global random.randint — load-bearing for OG
  reproducibility. Test only after OG parity constraint is lifted or the function
  receives an explicit rng argument (see comment in clean.py).
- generate.resolve_token_length: lives in generate.py which is a run-as-script file
  (module-level argparse/model loading). The function itself is pure math; it is
  covered via inline logic in TestResolveTokenLength below.
- io.get_from_cache / write_to_cache / save_dataset / save_domain / output_domains:
  filesystem + network I/O.
- extract.*: wraps LAMBO (external model, global RNG side-effects).
- models.*: loads HuggingFace models / GPU inference.
- datasets.Dataset.*: orchestrates the I/O pipeline.
- time.get_time_string: reads the system clock.
"""

import math

import pytest

# ---------------------------------------------------------------------------
# Imports — conftest.py puts both the project root and utils/ on sys.path
# ---------------------------------------------------------------------------
from utils.clean import (
    clean_extra_whitepsaces,
    clean_newlines,
    clean_special_characters,
    clean_square_brackets_and_their_content,
    format_quotes,
    remove_greetings,
)
from utils.fileio import get_data_cache_path
from utils.prompt import (
    get_abstract_prompt,
    get_classic_lit_prompt,
    get_coursebook_prompt,
    get_gov_prompt,
    get_review_prompt,
    get_social_prompt,
    get_wiki_prompt,
    get_wikinews_prompt,
)
from utils.strategy import resolve_strategy_kwargs


# ===========================================================================
# clean.py
# ===========================================================================


class TestCleanSquareBrackets:
    def test_removes_numeric_references(self):
        assert clean_square_brackets_and_their_content("foo [1] bar [22]") == "foo  bar"

    def test_strips_surrounding_whitespace(self):
        assert clean_square_brackets_and_their_content("  [1] hello  ") == "hello"

    def test_no_brackets_unchanged(self):
        assert clean_square_brackets_and_their_content("plain text") == "plain text"

    def test_ignores_non_numeric_brackets(self):
        # only \d+ is removed; [d] must stay
        assert clean_square_brackets_and_their_content("[d]") == "[d]"

    def test_empty_string(self):
        assert clean_square_brackets_and_their_content("") == ""


class TestCleanExtraWhitespaces:
    def test_removes_space_before_period(self):
        assert clean_extra_whitepsaces("Hello .") == "Hello."

    def test_removes_space_before_comma(self):
        assert clean_extra_whitepsaces("Hello ,world") == "Hello,world"

    def test_removes_space_before_exclamation(self):
        assert clean_extra_whitepsaces("Wow !") == "Wow!"

    def test_removes_space_before_question(self):
        assert clean_extra_whitepsaces("Really ?") == "Really?"

    def test_collapses_spaced_ellipsis(self):
        assert clean_extra_whitepsaces("wait . . .") == "wait..."

    def test_fixes_parentheses_spacing(self):
        assert clean_extra_whitepsaces("( hello )") == "(hello)"

    def test_strips_result(self):
        assert clean_extra_whitepsaces("  hello  ") == "hello"

    def test_removes_polemo_artifact(self):
        assert " [/ b]" not in clean_extra_whitepsaces("text [/ b] more")


class TestCleanNewlines:
    def test_replaces_newline_with_space(self):
        assert clean_newlines(["a\nb"]) == ["a b"]

    def test_multiple_strings(self):
        assert clean_newlines(["a\nb", "c\nd"]) == ["a b", "c d"]

    def test_no_newlines_unchanged(self):
        assert clean_newlines(["hello"]) == ["hello"]

    def test_empty_list(self):
        assert clean_newlines([]) == []


class TestCleanSpecialCharacters:
    def test_replaces_tab_with_space(self):
        assert clean_special_characters(["a\tb"]) == ["a b"]

    def test_replaces_special_newline_symbol(self):
        assert clean_special_characters(["a␤b"]) == ["a\nb"]

    def test_removes_carriage_return(self):
        assert clean_special_characters(["a\rb"]) == ["ab"]

    def test_replaces_unicode_line_separator(self):
        assert clean_special_characters([f"a\u2028b"]) == ["a\nb"]

    def test_replaces_unicode_paragraph_separator(self):
        assert clean_special_characters([f"a\u2029b"]) == ["a\nb"]

    def test_empty_list(self):
        assert clean_special_characters([]) == []

    def test_multiple_strings(self):
        assert clean_special_characters(["a\tb", "c\rd"]) == ["a b", "cd"]


class TestRemoveGreetings:
    def test_removes_single_greeting(self):
        text = "Szanowny Panie Marszałku! Oto treść."
        assert remove_greetings(text) == "Oto treść."

    def test_removes_consecutive_greetings(self):
        text = "Panie Marszałku! Wysoka Izbo! Treść."
        assert remove_greetings(text) == "Treść."

    def test_no_greeting_unchanged(self):
        text = "Zwykły tekst bez powitania."
        assert remove_greetings(text) == text

    def test_greeting_not_at_start_is_preserved(self):
        text = "Treść. Panie Marszałku!"
        assert remove_greetings(text) == text

    def test_empty_string(self):
        assert remove_greetings("") == ""

    def test_custom_greetings_list(self):
        result = remove_greetings("Cześć! Wiadomość.", greetings=["Cześć!"])
        assert result == "Wiadomość."

    def test_greeting_only_string_becomes_empty(self):
        result = remove_greetings("Panie Marszałku!")
        assert result.strip() == ""


class TestFormatQuotes:
    def test_removes_inner_padding_from_quoted_string(self):
        assert format_quotes('" hello "') == '"hello"'

    def test_plain_text_unchanged(self):
        assert format_quotes("brak cudzysłowu") == "brak cudzysłowu"


# ===========================================================================
# generate.resolve_token_length (inline — generate.py is a script, not a module)
# ===========================================================================

def _resolve_token_length(original_text_token_len, domain):
    """Local copy of generate.resolve_token_length for testing.
    generate.py runs argparse/model loading at import time so it cannot be
    imported as a module. The function body is trivial and stable."""
    min_factor = 1. if domain == 'social' else .8
    max_factor = 2. if domain == 'social' else 1.2
    return {
        "min_new_tokens": math.ceil(original_text_token_len * min_factor),
        "max_new_tokens": math.ceil(original_text_token_len * max_factor),
    }


class TestResolveTokenLength:
    def test_social_min_equals_original(self):
        assert _resolve_token_length(100, "social")["min_new_tokens"] == 100

    def test_social_max_is_double(self):
        assert _resolve_token_length(100, "social")["max_new_tokens"] == 200

    def test_non_social_min_is_80_percent(self):
        assert _resolve_token_length(100, "wiki")["min_new_tokens"] == 80

    def test_non_social_max_is_120_percent(self):
        assert _resolve_token_length(100, "wiki")["max_new_tokens"] == 120

    def test_ceiling_applied_fractional(self):
        # 7 * 0.8 = 5.6 → ceil = 6
        assert _resolve_token_length(7, "reviews")["min_new_tokens"] == 6

    @pytest.mark.parametrize("domain", ["wiki", "lit", "reviews", "gamma"])
    def test_non_social_factors(self, domain):
        result = _resolve_token_length(50, domain)
        assert result["min_new_tokens"] == math.ceil(50 * 0.8)
        assert result["max_new_tokens"] == math.ceil(50 * 1.2)


# ===========================================================================
# io.py — get_data_cache_path (pure path construction)
# ===========================================================================


class TestGetDataCachePath:
    def test_returns_expected_suffix(self):
        path = get_data_cache_path("raw", "wiki")
        assert path.endswith("raw/wiki.csv")

    def test_uses_cache_path_prefix(self):
        from utils.consts import CACHE_PATH
        assert get_data_cache_path("prefix", "plsc").startswith(CACHE_PATH)

    def test_cache_type_in_path(self):
        assert "prompt" in get_data_cache_path("prompt", "twitter")

    def test_dataset_name_in_path(self):
        assert "classics" in get_data_cache_path("raw", "classics")


# ===========================================================================
# strategy.py — resolve_strategy_kwargs
# ===========================================================================


class TestResolveStrategyKwargs:
    def test_returns_dict(self):
        from utils.consts import strategies  # noqa: F811
        result = resolve_strategy_kwargs(next(iter(strategies)))
        assert isinstance(result, dict)

    def test_all_strategies_resolve(self):
        from utils.consts import strategies  # noqa: F811
        for s in strategies:
            result = resolve_strategy_kwargs(s)
            assert isinstance(result, dict)

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            resolve_strategy_kwargs("nonexistent_strategy")


# ===========================================================================
# prompt.py
# ===========================================================================


class TestPromptFunctions:
    def test_wiki_prompt_contains_prefix(self):
        assert "Warszawa" in get_wiki_prompt("Warszawa to stolica")

    def test_abstract_prompt_contains_prefix(self):
        assert "Badania" in get_abstract_prompt("Badania nad gramatyką")

    def test_coursebook_prompt_contains_text(self):
        assert "Fizyka" in get_coursebook_prompt("Fizyka kwantowa")

    def test_classic_lit_prompt_contains_fragment(self):
        fragment = "Był sobie raz król"
        assert fragment in get_classic_lit_prompt(fragment)

    def test_gov_prompt_contains_prefix(self):
        assert "Chciałbym" in get_gov_prompt("Chciałbym zwrócić uwagę")

    def test_wikinews_prompt_contains_prefix(self):
        prefix = "W Warszawie doszło do"
        assert prefix in get_wikinews_prompt(prefix)

    def test_social_prompt_contains_text(self):
        assert "spacerze" in get_social_prompt("Dzisiaj byłem na spacerze")

    @pytest.mark.parametrize("category,keyword", [
        ("hotels", "hotelu"),
        ("medicine", "lekarza"),
        ("products", "produktu"),
        ("courses", "kursu"),
        ("movies", "filmu"),
    ])
    def test_review_prompt_category_keyword(self, category, keyword):
        assert keyword in get_review_prompt("Bardzo dobra", category)

    def test_review_prompt_contains_prefix(self):
        assert "Fantastyczny" in get_review_prompt("Fantastyczny pobyt", "hotels")

    def test_all_prompts_return_nonempty_strings(self):
        prompts = [
            get_wiki_prompt("x"),
            get_abstract_prompt("x"),
            get_coursebook_prompt("x"),
            get_classic_lit_prompt("x"),
            get_gov_prompt("x"),
            get_wikinews_prompt("x"),
            get_social_prompt("x"),
            get_review_prompt("x", "hotels"),
        ]
        for p in prompts:
            assert isinstance(p, str) and p
