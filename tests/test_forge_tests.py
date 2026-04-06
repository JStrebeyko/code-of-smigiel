"""Tests for postprocessing/forge_tests.py.

Pure-function unit tests: split_in, interleave_lists.
Integration tests: load/write round-trip and full TEST A/B assembly pipeline,
using the same structural checks as compare_postprocessing in results_consistency.ipynb
(file existence, line-count synchronisation, partition completeness, no data loss).
"""

import pathlib

import pytest

from postprocessing.forge_tests import (
    interleave_lists,
    load_txt_key_meta,
    split_in,
    write_txt_key_meta,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors compare_postprocessing logic from results_consistency.ipynb)
# ---------------------------------------------------------------------------

def _write_source(directory: pathlib.Path, tuples):
    """Write (text, key, meta) tuples to start.{txt,key,meta}."""
    write_txt_key_meta(tuples, directory)


def _make_tuples(n, prefix="item"):
    """Synthetic (text, key, meta) tuples — each line has a unique, traceable value."""
    return [
        (f"{prefix}_{i}_text\n", f"{i % 2}\n", f"hash_{prefix}_{i}\tsrc\tgreedy\n")
        for i in range(n)
    ]


def assert_dir_is_valid(directory: pathlib.Path, expected_lines: int):
    """Structural check: all three files exist and have the expected line count."""
    for ext in ("txt", "key", "meta"):
        f = directory / f"start.{ext}"
        assert f.exists(), f"start.{ext} missing in {directory}"
        lines = f.read_text().splitlines()
        assert len(lines) == expected_lines, (
            f"start.{ext}: expected {expected_lines} lines, got {len(lines)}"
        )


# ===========================================================================
# split_in
# ===========================================================================


class TestSplitIn:
    def test_single_bucket_returns_all(self):
        assert split_in(1, [1, 2, 3]) == [[1, 2, 3]]

    def test_two_buckets_even(self):
        result = split_in(2, [0, 1, 2, 3])
        assert result == [[0, 2], [1, 3]]

    def test_two_buckets_odd_length(self):
        result = split_in(2, [0, 1, 2, 3, 4])
        assert result == [[0, 2, 4], [1, 3]]

    def test_three_buckets(self):
        result = split_in(3, list(range(9)))
        assert result == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

    def test_three_buckets_non_divisible(self):
        result = split_in(3, list(range(10)))
        # 10 elements: buckets get 4, 3, 3
        assert len(result[0]) == 4
        assert len(result[1]) == 3
        assert len(result[2]) == 3

    def test_preserves_all_elements(self):
        items = list(range(20))
        buckets = split_in(4, items)
        assert sorted(e for b in buckets for e in b) == items

    def test_empty_input(self):
        assert split_in(3, []) == []

    def test_fewer_items_than_buckets(self):
        result = split_in(5, [10, 20])
        assert result == [[10], [20]]

    def test_round_robin_order(self):
        # Element i must land in bucket i % n
        items = ['a', 'b', 'c', 'd', 'e', 'f']
        result = split_in(3, items)
        assert result[0] == ['a', 'd']
        assert result[1] == ['b', 'e']
        assert result[2] == ['c', 'f']

    def test_returns_correct_number_of_buckets(self):
        result = split_in(4, list(range(12)))
        assert len(result) == 4

    def test_alpha_beta_gamma_split_sizes(self):
        # Replicate OG split ratios
        alpha = list(range(4505))
        beta  = list(range(4356))
        gamma = list(range(19914))

        alpha_a, alpha_b        = split_in(2, alpha)
        beta_thirds             = split_in(3, beta)
        gamma_thirds            = split_in(3, gamma)

        assert len(alpha_a) == 2253
        assert len(alpha_b) == 2252
        assert len(beta_thirds[0]) == 1452
        assert len(beta_thirds[1]) + len(beta_thirds[2]) == 2904
        assert len(gamma_thirds[0]) == 6638
        assert len(gamma_thirds[1]) + len(gamma_thirds[2]) == 13276


# ===========================================================================
# interleave_lists
# ===========================================================================


class TestInterleaveLists:
    def test_empty_input(self):
        assert interleave_lists([]) == []

    def test_all_empty_lists(self):
        assert interleave_lists([[], [], []]) == []

    def test_single_list_passthrough(self):
        assert interleave_lists([[1, 2, 3]]) == [1, 2, 3]

    def test_preserves_all_elements(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = interleave_lists([a, b])
        assert sorted(result) == [1, 2, 3, 4, 5, 6]

    def test_no_consecutive_same_source_equal_lists(self):
        a = list(range(5))
        b = list(range(10, 15))
        result = interleave_lists([a, b])
        for i in range(len(result) - 1):
            assert not (result[i] in a and result[i + 1] in a), \
                f"Two consecutive elements from list a at positions {i}, {i+1}"
            assert not (result[i] in b and result[i + 1] in b), \
                f"Two consecutive elements from list b at positions {i}, {i+1}"

    def test_three_equal_lists_no_consecutive(self):
        a = [0, 3, 6]
        b = [1, 4, 7]
        c = [2, 5, 8]
        result = interleave_lists([a, b, c])
        assert len(result) == 9
        for i in range(len(result) - 1):
            # Same-source consecutive check via set membership
            assert not (result[i] in a and result[i + 1] in a)
            assert not (result[i] in b and result[i + 1] in b)
            assert not (result[i] in c and result[i + 1] in c)

    def test_dominant_list_tail_appended(self):
        # One list is much larger — its tail will appear at the end
        big = list(range(100))
        small = [200, 201]
        result = interleave_lists([big, small])
        assert len(result) == 102
        assert sorted(result) == sorted(big + small)

    def test_empty_list_ignored(self):
        result = interleave_lists([[1, 2, 3], [], [4, 5, 6]])
        assert sorted(result) == [1, 2, 3, 4, 5, 6]

    def test_output_length_matches_total_input(self):
        lists = [list(range(i * 10, i * 10 + 7)) for i in range(5)]
        result = interleave_lists(lists)
        assert len(result) == sum(len(l) for l in lists)

    def test_deterministic(self):
        a = list(range(5))
        b = list(range(10, 15))
        assert interleave_lists([a, b]) == interleave_lists([a, b])

    def test_forge_scale(self):
        # Rough scale of actual TEST A assembly (alpha½ + beta⅓ + gamma⅓)
        alpha_a = list(range(2253))
        beta_a  = list(range(10000, 11452))
        gamma_a = list(range(20000, 26638))
        result = interleave_lists([alpha_a, beta_a, gamma_a])
        assert len(result) == 2253 + 1452 + 6638
        assert set(result) == set(alpha_a) | set(beta_a) | set(gamma_a)


# ===========================================================================
# I/O round-trip
# ===========================================================================


class TestLoadWriteRoundTrip:
    def test_write_then_load_identity(self, tmp_path):
        original = _make_tuples(20)
        _write_source(tmp_path / "src", original)
        loaded = load_txt_key_meta(tmp_path / "src")
        assert loaded == original

    def test_files_created(self, tmp_path):
        _write_source(tmp_path / "src", _make_tuples(5))
        assert_dir_is_valid(tmp_path / "src", expected_lines=5)

    def test_empty_source(self, tmp_path):
        _write_source(tmp_path / "empty", [])
        loaded = load_txt_key_meta(tmp_path / "empty")
        assert loaded == []


# ===========================================================================
# Full TEST A / TEST B assembly pipeline
# ===========================================================================


class TestForgeTestsPipeline:
    """Integration tests mirroring compare_postprocessing checks from
    results_consistency.ipynb: files exist, line counts are correct,
    txt/key/meta are synchronised, and alpha+beta+gamma are partitioned
    exactly between TEST A and TEST B with no data loss."""

    @pytest.fixture()
    def sources(self, tmp_path):
        """Write synthetic alpha (12 items), beta (9 items), gamma (18 items)."""
        alpha = _make_tuples(12, "alpha")
        beta  = _make_tuples(9,  "beta")
        gamma = _make_tuples(18, "gamma")
        _write_source(tmp_path / "alpha", alpha)
        _write_source(tmp_path / "beta",  beta)
        _write_source(tmp_path / "gamma", gamma)
        return tmp_path, alpha, beta, gamma

    def _run_pipeline(self, tmp_path, alpha, beta, gamma):
        alpha_a, alpha_b          = split_in(2, alpha)
        beta_a, *beta_b_parts     = split_in(3, beta)
        gamma_a, *gamma_b_parts   = split_in(3, gamma)

        test_a = interleave_lists([alpha_a, beta_a, gamma_a])
        test_b = interleave_lists([alpha_b,
                                   beta_b_parts[0]  + beta_b_parts[1],
                                   gamma_b_parts[0] + gamma_b_parts[1]])

        out_a = tmp_path / "test_A"
        out_b = tmp_path / "test_B"
        write_txt_key_meta(test_a, out_a)
        write_txt_key_meta(test_b, out_b)
        return out_a, out_b, test_a, test_b

    def test_output_files_exist(self, sources):
        tmp_path, alpha, beta, gamma = sources
        out_a, out_b, *_ = self._run_pipeline(tmp_path, alpha, beta, gamma)
        for d in (out_a, out_b):
            for ext in ("txt", "key", "meta"):
                assert (d / f"start.{ext}").exists()

    def test_files_are_line_synchronised(self, sources):
        """txt, key, and meta must have identical line counts (mirror of
        compare_postprocessing line-count check in results_consistency.ipynb)."""
        tmp_path, alpha, beta, gamma = sources
        out_a, out_b, test_a, test_b = self._run_pipeline(tmp_path, alpha, beta, gamma)
        assert_dir_is_valid(out_a, expected_lines=len(test_a))
        assert_dir_is_valid(out_b, expected_lines=len(test_b))

    def test_no_data_loss(self, sources):
        """Every tuple from all three sources must appear in exactly one output."""
        tmp_path, alpha, beta, gamma = sources
        _, _, test_a, test_b = self._run_pipeline(tmp_path, alpha, beta, gamma)
        all_input  = set(map(tuple, alpha + beta + gamma))
        all_output = set(map(tuple, test_a + test_b))
        assert all_input == all_output

    def test_partition_is_disjoint(self, sources):
        """TEST A and TEST B must not share any tuple."""
        tmp_path, alpha, beta, gamma = sources
        _, _, test_a, test_b = self._run_pipeline(tmp_path, alpha, beta, gamma)
        assert set(map(tuple, test_a)).isdisjoint(set(map(tuple, test_b)))

    def test_split_sizes(self, sources):
        """alpha→halves, beta/gamma→thirds with 1/3 to A and 2/3 to B."""
        tmp_path, alpha, beta, gamma = sources
        _, _, test_a, test_b = self._run_pipeline(tmp_path, alpha, beta, gamma)
        # alpha: 12 → 6+6; beta: 9 → 3+6; gamma: 18 → 6+12
        assert len(test_a) == 6 + 3 + 6   # 15
        assert len(test_b) == 6 + 6 + 12  # 24

    def test_loaded_outputs_match_written(self, sources):
        """Round-trip: what write_txt_key_meta writes, load_txt_key_meta reads back."""
        tmp_path, alpha, beta, gamma = sources
        out_a, out_b, test_a, test_b = self._run_pipeline(tmp_path, alpha, beta, gamma)
        assert load_txt_key_meta(out_a) == test_a
        assert load_txt_key_meta(out_b) == test_b
