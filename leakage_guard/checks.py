"""Look-ahead leakage checks for time-series forecasting pipelines.

Standard library only. Every check takes *your* function, runs it on perturbed
copies of the same data, and raises ``CheckFailure`` if the answer for the past
changes when only the future (or an unknown label) was touched.

Row convention: a list of dicts sorted by ``time_key`` (ties allowed). A
feature function maps ``rows -> one feature entry per row`` (numbers, lists, or
nested lists of numbers).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

__all__ = [
    "CheckFailure", "prefix_invariance", "future_mutation_invariance",
    "immature_label_invariance", "deterministic_rerun",
    "failure_produces_no_output", "permuted",
]


class CheckFailure(AssertionError):
    """A pipeline violated a leakage check."""


def _same(a, b, tol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        if a != a and b != b:          # both NaN: treat as equal
            return True
        try:
            return a == b or abs(a - b) <= tol
        except TypeError:
            return False
    return a == b


def _require_sorted(rows, time_key):
    times = [r[time_key] for r in rows]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("rows must be sorted by %r" % time_key)


def _features(feature_fn, rows):
    out = list(feature_fn(copy.deepcopy(rows)))
    if len(out) != len(rows):
        raise CheckFailure("feature_fn must return exactly one entry per row "
                           "(got %d for %d rows)" % (len(out), len(rows)))
    return out


def prefix_invariance(feature_fn, rows, cutoffs=None, time_key="t", tol=0.0):
    """Features of old rows must not change when later rows are removed.

    Catches centered windows, normalisation fitted on the whole series, and any
    other statistic that peeks past the row's own time. Returns the number of
    cutoffs checked.
    """
    rows = list(rows)
    _require_sorted(rows, time_key)
    full = _features(feature_fn, rows)
    times = sorted({r[time_key] for r in rows})
    if cutoffs is None:
        cutoffs = times[:-1]
    cutoffs = list(cutoffs)
    if not cutoffs:
        raise ValueError("nothing to check: need at least two distinct times")
    for cutoff in cutoffs:
        k = sum(1 for r in rows if r[time_key] <= cutoff)
        part = _features(feature_fn, rows[:k])
        for i in range(k):
            if not _same(part[i], full[i], tol):
                raise CheckFailure(
                    "row %d (t=%r) changes when data after cutoff %r is removed: "
                    "%r != %r" % (i, rows[i][time_key], cutoff, part[i], full[i]))
    return len(cutoffs)


def future_mutation_invariance(feature_fn, rows, cutoff, mutate,
                               time_key="t", tol=0.0):
    """Wildly changing rows after ``cutoff`` must not change earlier features.

    ``mutate(row_copy) -> row`` is applied to every row with time > cutoff.
    Returns the number of mutated rows.
    """
    rows = list(rows)
    _require_sorted(rows, time_key)
    base = _features(feature_fn, rows)
    changed, n = [], 0
    for r in rows:
        if r[time_key] > cutoff:
            changed.append(mutate(dict(r)))
            n += 1
        else:
            changed.append(dict(r))
    if n == 0:
        raise ValueError("cutoff leaves no future rows to mutate")
    other = _features(feature_fn, changed)
    for i, r in enumerate(rows):
        if r[time_key] <= cutoff and not _same(base[i], other[i], tol):
            raise CheckFailure(
                "row %d (t=%r) changes when rows after cutoff %r are altered: "
                "%r != %r" % (i, r[time_key], cutoff, base[i], other[i]))
    return n


def immature_label_invariance(fit_predict, rows, cutoff, label_time_key,
                              label_keys, poison, time_key="t", tol=0.0):
    """Labels that are not yet known at ``cutoff`` must not affect the result.

    ``fit_predict(rows, cutoff) -> result``. Rows whose ``label_time_key`` is
    later than ``cutoff`` get their ``label_keys`` replaced by ``poison``.
    Returns the number of poisoned rows.
    """
    rows = list(rows)
    base = fit_predict(copy.deepcopy(rows), cutoff)
    changed, n = [], 0
    for r in rows:
        r2 = dict(r)
        if r2[label_time_key] > cutoff:
            for key in label_keys:
                r2[key] = poison
            n += 1
        changed.append(r2)
    if n == 0:
        raise ValueError("no immature labels at this cutoff; pick an earlier one")
    other = fit_predict(copy.deepcopy(changed), cutoff)
    if not _same(base, other, tol):
        raise CheckFailure(
            "result at cutoff %r depends on %d labels that are not yet known: "
            "%r != %r" % (cutoff, n, base, other))
    return n


def deterministic_rerun(run, runs=2):
    """``run()`` must return identical bytes (or JSON-able data) every time.

    Returns the SHA-256 shared by all runs. NaN is rejected in JSON data.
    """
    if runs < 2:
        raise ValueError("runs must be at least 2")
    digests = []
    for _ in range(runs):
        out = run()
        blob = out if isinstance(out, bytes) else json.dumps(
            out, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        digests.append(hashlib.sha256(blob).hexdigest())
    if len(set(digests)) != 1:
        raise CheckFailure("reruns differ: %s" % ", ".join(d[:12] for d in digests))
    return digests[0]


def failure_produces_no_output(run, output_path):
    """A pipeline forced to fail must not leave (or reuse) an output file.

    ``run()`` should raise. The output path must not exist before or after.
    """
    path = Path(output_path)
    if path.exists():
        raise ValueError("%s already exists; start from a clean directory" % path)
    try:
        run()
    except Exception:
        pass
    else:
        raise CheckFailure("run() was expected to fail but succeeded")
    if path.exists():
        raise CheckFailure("%s exists after a failed run" % path)


def permuted(values, seed):
    """Reproducible permutation with the same marginals (negative control)."""
    out = list(values)
    random.Random(seed).shuffle(out)
    return out
