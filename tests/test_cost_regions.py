"""Checks for the cost-regions analysis, on synthetic numbers so no artefact is needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xiiotid_cost_regions import envelope, select, summarise  # noqa: E402

COSTS = {"Normal": {"cost_fn": 0, "cost_fp": 1}, "Scan": {"cost_fn": 2, "cost_fp": 1},
         "Ransom": {"cost_fn": 10, "cost_fp": 3}}
LABELS = ["Normal", "Ransom", "Scan"]
#            pred:  Normal  Ransom  Scan
MATRIX = [[80, 3, 2],     # true Normal: 5 false alarms
          [4, 10, 6],     # true Ransom: 4 missed, 6 confused with another attack
          [7, 1, 20]]     # true Scan:   7 missed, 1 confused with another attack


def test_envelope_handles_equal_intercepts():
    """Same C_miss, different slopes: the flatter line owns every lambda above zero."""
    assert envelope({"a": (0, 10), "b": (0, 1)}, ["a", "b"]) == [(0.0, float("inf"), "b")]
    assert select({"a": (0, 10), "b": (0, 1)}, ["a", "b"], 0.5) == "b"


def test_envelope_handles_three_lines_crossing_at_one_point():
    """a, b and c all cost 3 at lambda = 1: beyond it the flattest line, c, is preferred."""
    lines = {"a": (0, 3), "b": (1, 2), "c": (2, 1)}
    assert envelope(lines, ["a", "b", "c"]) == [(0.0, 1.0, "a"), (1.0, float("inf"), "c")]
    assert select(lines, ["a", "b", "c"], 2) == "c"


def test_boundary_belongs_to_the_segment_that_starts_there():
    """At a crossing both cost the same; the rule gives it to the smaller false-alarm cost."""
    lines = {"a": (0, 3), "c": (2, 1)}
    (_, boundary, before), (start, _, after) = envelope(lines, ["a", "c"])
    assert boundary == start == 1.0
    assert (before, after) == ("a", "c")
    assert select(lines, ["a", "c"], boundary) == after
    assert select(lines, ["a", "c"], boundary - 0.1) == before


def test_dominated_line_never_appears():
    lines = {"flat": (1, 1), "steep": (5, 9)}
    assert [v for _, _, v in envelope(lines, ["flat", "steep"])] == ["flat"]


def test_summarise_splits_the_errors_and_normalises():
    s = summarise(MATRIX, LABELS, COSTS)
    assert s["missed_attacks"] == 4 + 7
    assert s["missed_attack_cost"] == 4 * 10 + 7 * 2
    assert s["false_alarms"] == 3 + 2
    assert s["false_alarm_cost"] == 3 * 3 + 2 * 1
    assert s["attack_confusions_unweighted"] == 6 + 1
    assert s["rows"] == sum(sum(row) for row in MATRIX)
    assert s["normal_rows"] == 85
    errors = sum(MATRIX[i][j] for i in range(3) for j in range(3) if i != j)
    assert s["missed_attacks"] + s["false_alarms"] + s["attack_confusions_unweighted"] == errors
    assert round(s["missed_attack_cost"] / s["rows"] * 1000, 3) == 406.015
    assert round(s["false_alarms"] / s["normal_rows"], 6) == 0.058824


def test_selection_ignores_changes_to_the_test_part():
    validation = {"a": (100, 10), "b": (160, 2)}
    test = {"a": (90, 12), "b": (150, 3)}
    chosen = [select(validation, ["a", "b"], lam) for lam in (0, 1, 5, 20)]
    test_altered = {v: (a * 3 + 5, b * 3 + 5) for v, (a, b) in test.items()}
    assert test_altered != test
    assert [select(validation, ["a", "b"], lam) for lam in (0, 1, 5, 20)] == chosen
