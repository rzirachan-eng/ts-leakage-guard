"""Tiny synthetic pipelines: one correct and one leaky version of each pattern.

All data are generated here; nothing is read from disk.
"""
import math
import random


def make_rows(n=30, horizon=2):
    """Weekly series; the label of row i is the value ``horizon`` steps later."""
    values = [100.0 + 10.0 * math.sin(i / 3.0) + i for i in range(n + horizon)]
    return [{"t": i, "value": values[i], "label_t": i + horizon,
             "label": values[i + horizon]} for i in range(n)]


# --- feature functions -------------------------------------------------------

def trailing_mean(rows, window=3):
    """Correct: uses only the current and earlier rows."""
    out = []
    for i in range(len(rows)):
        chunk = [r["value"] for r in rows[max(0, i - window + 1):i + 1]]
        out.append([sum(chunk) / len(chunk)])
    return out


def centered_mean(rows, window=3):
    """Leaky: a centred window reads the next row."""
    half = window // 2
    out = []
    for i in range(len(rows)):
        chunk = [r["value"] for r in rows[max(0, i - half):i + half + 1]]
        out.append([sum(chunk) / len(chunk)])
    return out


def zscore_whole_series(rows):
    """Leaky: normalisation statistics are fitted on the whole series."""
    values = [r["value"] for r in rows]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var) or 1.0
    return [[(v - mean) / std] for v in values]


def zscore_expanding(rows):
    """Correct: statistics use only rows up to the current one."""
    out = []
    for i in range(len(rows)):
        seen = [r["value"] for r in rows[:i + 1]]
        mean = sum(seen) / len(seen)
        std = math.sqrt(sum((v - mean) ** 2 for v in seen) / len(seen)) or 1.0
        out.append([(rows[i]["value"] - mean) / std])
    return out


# --- fit/predict functions ---------------------------------------------------

def fit_predict_mature_only(rows, cutoff):
    """Correct: trains only on labels already known at the cutoff."""
    known = [r["label"] for r in rows if r["label_t"] <= cutoff]
    return sum(known) / len(known) if known else 0.0


def fit_predict_all_labels(rows, cutoff):
    """Leaky: trains on every label, including ones that mature after cutoff."""
    labels = [r["label"] for r in rows if r["t"] <= cutoff]
    return sum(labels) / len(labels)


# --- reruns and failures -----------------------------------------------------

def seeded_run():
    rng = random.Random(7)
    return [round(rng.random(), 6) for _ in range(5)]


def unseeded_run():
    return [random.random() for _ in range(5)]


def fails_before_writing(path):
    raise RuntimeError("forced failure")


def writes_partial_then_fails(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("partial predictions\n")
    raise RuntimeError("forced failure")
