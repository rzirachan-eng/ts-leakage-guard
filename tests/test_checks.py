import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import pipelines as p  # noqa: E402
from leakage_guard import (CheckFailure, deterministic_rerun,  # noqa: E402
                           failure_produces_no_output,
                           future_mutation_invariance,
                           immature_label_invariance, permuted,
                           prefix_invariance)


def spike(row):
    row["value"] = 1e9
    return row


class PrefixInvarianceTests(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_rows()

    def test_causal_features_pass(self):
        for fn in (p.trailing_mean, p.zscore_expanding):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(prefix_invariance(fn, self.rows), len(self.rows) - 1)

    def test_centered_window_is_caught(self):
        with self.assertRaises(CheckFailure) as ctx:
            prefix_invariance(p.centered_mean, self.rows)
        self.assertIn("row 0", str(ctx.exception))

    def test_whole_series_normalisation_is_caught(self):
        with self.assertRaises(CheckFailure):
            prefix_invariance(p.zscore_whole_series, self.rows)

    def test_explicit_cutoffs(self):
        self.assertEqual(prefix_invariance(p.trailing_mean, self.rows, cutoffs=[5, 12]), 2)

    def test_unsorted_rows_rejected(self):
        rows = list(reversed(self.rows))
        with self.assertRaises(ValueError):
            prefix_invariance(p.trailing_mean, rows)

    def test_wrong_output_length_is_a_failure(self):
        with self.assertRaises(CheckFailure):
            prefix_invariance(lambda rows: [[0.0]], self.rows)

    def test_single_time_has_nothing_to_check(self):
        with self.assertRaises(ValueError):
            prefix_invariance(p.trailing_mean, self.rows[:1])

    def test_input_rows_are_not_modified(self):
        import copy
        before = copy.deepcopy(self.rows)

        def mutating(rows):
            for r in rows:
                r["value"] = 0.0
            return [[0.0] for _ in rows]

        prefix_invariance(mutating, self.rows)
        self.assertEqual(self.rows, before)

    def test_tolerance(self):
        def noisy(rows):
            return [[r["value"] + (1e-12 if len(rows) % 2 else 0.0)] for r in rows]

        with self.assertRaises(CheckFailure):
            prefix_invariance(noisy, self.rows, cutoffs=[10])
        prefix_invariance(noisy, self.rows, cutoffs=[10], tol=1e-9)

    def test_nan_equals_nan(self):
        prefix_invariance(lambda rows: [[math.nan] for _ in rows], self.rows)


class FutureMutationTests(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_rows()

    def test_causal_features_pass(self):
        for fn in (p.trailing_mean, p.zscore_expanding):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(
                    future_mutation_invariance(fn, self.rows, 20, spike), 9)

    def test_leaks_are_caught(self):
        for fn in (p.centered_mean, p.zscore_whole_series):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(CheckFailure):
                    future_mutation_invariance(fn, self.rows, 20, spike)

    def test_cutoff_without_future_rows(self):
        with self.assertRaises(ValueError):
            future_mutation_invariance(p.trailing_mean, self.rows, 10**6, spike)


class ImmatureLabelTests(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_rows()

    def test_mature_only_passes(self):
        n = immature_label_invariance(p.fit_predict_mature_only, self.rows, 20,
                                      "label_t", ["label"], 1e20)
        self.assertEqual(n, 11)

    def test_peeking_at_unknown_labels_is_caught(self):
        with self.assertRaises(CheckFailure) as ctx:
            immature_label_invariance(p.fit_predict_all_labels, self.rows, 20,
                                      "label_t", ["label"], 1e20)
        self.assertIn("not yet known", str(ctx.exception))

    def test_no_immature_labels(self):
        with self.assertRaises(ValueError):
            immature_label_invariance(p.fit_predict_mature_only, self.rows, 10**6,
                                      "label_t", ["label"], 1e20)


class DeterministicRerunTests(unittest.TestCase):
    def test_seeded_run_passes(self):
        digest = deterministic_rerun(p.seeded_run, runs=3)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_unseeded_run_is_caught(self):
        with self.assertRaises(CheckFailure):
            deterministic_rerun(p.unseeded_run)

    def test_bytes_output(self):
        self.assertEqual(deterministic_rerun(lambda: b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_dict_key_order_does_not_matter(self):
        flip = iter([{"a": 1, "b": 2}, {"b": 2, "a": 1}])
        deterministic_rerun(lambda: next(flip))

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_rerun(lambda: [math.nan])

    def test_needs_two_runs(self):
        with self.assertRaises(ValueError):
            deterministic_rerun(p.seeded_run, runs=1)


class FailureProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "predictions.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_failure_passes(self):
        failure_produces_no_output(lambda: p.fails_before_writing(self.out), self.out)

    def test_partial_output_is_caught(self):
        with self.assertRaises(CheckFailure):
            failure_produces_no_output(
                lambda: p.writes_partial_then_fails(self.out), self.out)

    def test_run_that_succeeds_is_caught(self):
        with self.assertRaises(CheckFailure):
            failure_produces_no_output(lambda: None, self.out)

    def test_stale_output_rejected_up_front(self):
        self.out.write_text("old", encoding="utf-8")
        with self.assertRaises(ValueError):
            failure_produces_no_output(lambda: p.fails_before_writing(self.out), self.out)


class PermutedTests(unittest.TestCase):
    def test_same_marginals_and_reproducible(self):
        values = list(range(20))
        a, b = permuted(values, 1), permuted(values, 1)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a), values)
        self.assertNotEqual(a, values)
        self.assertNotEqual(a, permuted(values, 2))

    def test_input_unchanged(self):
        values = [3, 1, 2]
        permuted(values, 0)
        self.assertEqual(values, [3, 1, 2])


class DemoTests(unittest.TestCase):
    def test_demo_runs_and_reports_both_outcomes(self):
        import contextlib
        import io
        import runpy
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(ROOT / "examples" / "demo.py"), run_name="__main__")
        text = buf.getvalue()
        self.assertEqual(text.count("LEAK  "), 7)
        self.assertEqual(text.count("PASS  "), 7)


if __name__ == "__main__":
    unittest.main()
