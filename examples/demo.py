"""Run every check against a correct and a leaky pipeline and print the verdicts.

    python examples/demo.py
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pipelines as p  # noqa: E402
from leakage_guard import (CheckFailure, deterministic_rerun,  # noqa: E402
                           failure_produces_no_output,
                           future_mutation_invariance,
                           immature_label_invariance, prefix_invariance)


def verdict(name, fn):
    try:
        fn()
        print("PASS  %s" % name)
    except CheckFailure as exc:
        print("LEAK  %s\n      -> %s" % (name, exc))


def main():
    rows = p.make_rows()

    def spike(row):
        row["value"] = 1e9
        return row

    for label, feat in [("trailing_mean", p.trailing_mean),
                        ("centered_mean", p.centered_mean),
                        ("zscore_expanding", p.zscore_expanding),
                        ("zscore_whole_series", p.zscore_whole_series)]:
        verdict("prefix_invariance / " + label,
                lambda f=feat: prefix_invariance(f, rows))
        verdict("future_mutation_invariance / " + label,
                lambda f=feat: future_mutation_invariance(f, rows, 20, spike))

    for label, fit in [("mature_only", p.fit_predict_mature_only),
                       ("all_labels", p.fit_predict_all_labels)]:
        verdict("immature_label_invariance / " + label,
                lambda f=fit: immature_label_invariance(
                    f, rows, 20, "label_t", ["label"], 1e20))

    verdict("deterministic_rerun / seeded",
            lambda: deterministic_rerun(p.seeded_run))
    verdict("deterministic_rerun / unseeded",
            lambda: deterministic_rerun(p.unseeded_run))

    with tempfile.TemporaryDirectory() as tmp:
        for label, fn in [("fails_before_writing", p.fails_before_writing),
                          ("writes_partial_then_fails", p.writes_partial_then_fails)]:
            out = Path(tmp) / (label + ".csv")
            verdict("failure_produces_no_output / " + label,
                    lambda fn=fn, out=out: failure_produces_no_output(
                        lambda: fn(out), out))


if __name__ == "__main__":
    main()
