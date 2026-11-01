"""Checks for the cost-robustness analysis: synthetic cases, plus the accepted matrices."""
import json
import os
import sys
from fractions import Fraction
from itertools import product

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from xiiotid_cost_robustness import (  # noqa: E402
    analyse, cost, extreme_vector, max_regret, merged, radius, regret_intervals, regret_lines, terms)

KEYS_ABC = {"a": (10, 0, 0), "b": (11, 0, 1), "c": (12, 0, 2)}
SWITCH = {"a": [10, 0], "b": [6, 5], "c": [0, 12]}  # with w0 = [1, 1]: costs 10, 11, 12


def load_real():
    def read(path):
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return json.load(f)
    protocol = read("configs/xiiotid_cost_robustness.json")
    costs = read(protocol["weights"])["classes"]
    metrics = {m: read(f"reports_xiiot/metrics_{m}.json") for m in protocol["pool"]}
    return metrics, costs, protocol


def selection(rows):
    return [(r["lo"], r["lo_closed"], r["hi"], r["hi_closed"], r["selected"]) for r in rows]


def test_delta_zero_gives_the_nominal_regret_and_choice():
    w0 = [1, 1]
    lines = {m: regret_lines(w0, SWITCH, m) for m in SWITCH}
    assert {m: max_regret(lines[m], Fraction(0)) for m in SWITCH} == {"a": 0, "b": 1, "c": 2}
    assert regret_intervals(w0, SWITCH, KEYS_ABC)[0]["selected"] == "a"


def test_radius_and_extreme_vector_on_a_known_pair():
    w0, e_m, e_k = [10, 1], [1, 15], [3, 0]
    a, b = terms(w0, e_m, e_k)
    assert (a, b) == (-5, 35)
    r = radius(a, b)
    assert r == Fraction(1, 7)
    extreme = extreme_vector(w0, e_m, e_k, r)
    assert extreme == [Fraction(60, 7), Fraction(8, 7)]
    assert cost(extreme, e_m) == cost(extreme, e_k)


def test_not_costlier_at_the_radius_and_strictly_better_only_below_it():
    a, b = terms([10, 1], [1, 15], [3, 0])
    r = radius(a, b)
    assert a + (r - Fraction(1, 100)) * b < 0
    assert a + r * b == 0
    assert a + (r + Fraction(1, 100)) * b > 0


def test_winner_changes_with_delta_and_the_tie_goes_to_the_nominal_cost():
    """R_a = -2 + 22 delta meets R_b = 1 + 9 delta at 3/13: a keeps that point by the tie-break."""
    rows = regret_intervals([1, 1], SWITCH, KEYS_ABC)
    boundary = Fraction(3, 13)
    assert selection(merged(rows)) == [(0, True, boundary, True, "a"), (boundary, False, 1, True, "b")]
    at_boundary = {m: max_regret(regret_lines([1, 1], SWITCH, m), boundary) for m in SWITCH}
    assert at_boundary["a"] == at_boundary["b"] == Fraction(40, 13)


def test_intervals_cover_every_delta_once_and_regret_is_continuous():
    rows = regret_intervals([1, 1], SWITCH, KEYS_ABC)
    assert (rows[0]["lo"], rows[0]["lo_closed"]) == (0, True)
    assert (rows[-1]["hi"], rows[-1]["hi_closed"]) == (1, True)
    for x, y in zip(rows, rows[1:]):
        assert x["hi"] == y["lo"] and x["hi_closed"] != y["lo_closed"]
        assert x["regret_hi"] == y["regret_lo"]


def test_identical_errors_are_equal_for_every_delta_and_tie_break_decides():
    a, b = terms([3, 2], [4, 5], [4, 5])
    assert (a, b) == (0, 0)
    assert radius(a, b) is None
    rows = regret_intervals([3, 2], {"x": [4, 5], "y": [4, 5]}, {"x": (22, -0.9, 1), "y": (22, -0.95, 0)})
    assert selection(rows) == [(0, True, 1, True, "y")]
    rows = regret_intervals([3, 2], {"x": [4, 5], "y": [4, 5]}, {"x": (22, -0.9, 1), "y": (22, -0.9, 0)})
    assert selection(rows) == [(0, True, 1, True, "y")]


def test_nominal_parity_with_different_errors_has_radius_zero_and_no_strict_win():
    a, b = terms([1, 1], [2, 0], [0, 2])
    assert (a, b) == (0, 4)
    assert radius(a, b) == 0
    assert a + Fraction(1, 10) * b > 0


def test_nominally_costlier_model_has_no_radius():
    assert radius(*terms([1], [5], [3])) is None


@pytest.mark.parametrize("w0, vectors", [
    ([10, 1], {"a": [1, 15], "b": [3, 0]}),
    ([1, 1], SWITCH),
    ([3, 1, 7], {"a": [4, 9, 0], "b": [0, 2, 5], "c": [6, 0, 1]}),
    ([2, 5, 4], {"a": [1, 1, 1], "b": [1, 1, 1], "c": [0, 3, 2]}),
])
@pytest.mark.parametrize("delta", [Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(1)])
def test_formula_matches_the_enumeration_of_the_vertices(w0, vectors, delta):
    vertices = [[w * (1 + s * delta) for w, s in zip(w0, signs)]
                for signs in product((-1, 1), repeat=len(w0))]
    for m in vectors:
        for k in vectors:
            a, b = terms(w0, vectors[m], vectors[k])
            assert a + delta * b == max(cost(w, vectors[m]) - cost(w, vectors[k]) for w in vertices)
        assert max_regret(regret_lines(w0, vectors, m), delta) == max(
            cost(w, vectors[m]) - min(cost(w, vectors[k]) for k in vectors) for w in vertices)


def test_reproduces_the_feasibility_values_of_the_card():
    result = analyse(*load_real())
    nominal = result["nominal"]
    assert (nominal["validation"]["lgbm"], nominal["validation"]["xgb"]) == (2758, 3222)
    assert (nominal["test"]["lgbm"], nominal["test"]["xgb"]) == (2794, 3014)
    pair = {part: next(r for r in result["pairwise"][part] if {r["m"], r["k"]} == {"lgbm", "xgb"})
            for part in ("validation", "test")}
    assert (pair["validation"]["m"], -pair["validation"]["a"], pair["validation"]["b"]) == ("lgbm", 464, 652)
    assert (pair["test"]["m"], -pair["test"]["a"], pair["test"]["b"]) == ("lgbm", 220, 856)
    assert {r["selected"] for r in result["regret"]["validation"]} == {"lgbm"}
    assert {r["selected"] for r in result["regret"]["test"]} == {"lgbm"}


def test_validation_selection_does_not_change_when_the_test_part_changes():
    metrics, costs, protocol = load_real()
    original = analyse(metrics, costs, protocol)
    tampered = {m: {**d, "confusion_matrix": {
        "labels": d["confusion_matrix"]["labels"],
        "matrix": [[(c * 3 + 5) * (1 + 7 * (m == "lgbm")) for c in row]
                   for row in d["confusion_matrix"]["matrix"]]}} for m, d in metrics.items()}
    altered = analyse(tampered, costs, protocol)
    assert altered["regret"]["test"] != original["regret"]["test"]
    assert altered["pairwise"]["test"] != original["pairwise"]["test"]
    for key in ("regret", "pairwise", "vectors", "nominal"):
        assert altered[key]["validation"] == original[key]["validation"]
    assert altered["nominal_choice"] == original["nominal_choice"]
    assert altered["macro_f1_choice"] == original["macro_f1_choice"]
