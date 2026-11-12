"""Checks for scripts/cost_selection.py: RISK-03 regression on the accepted matrices, and
synthetic cases for 34 classes, pools, label order, counts, supports, ties and normalisation."""
import json
import os
import random
import sys
from fractions import Fraction
from itertools import product

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cost_selection import (  # noqa: E402
    InputError, align, decide, errors, load, rate_vector, read_inputs, risk03_checks, scenario)
from xiiotid_cost_regions import sha256  # noqa: E402
from xiiotid_cost_robustness import cost  # noqa: E402

PROTOCOL = load(os.path.join(ROOT, "configs/common_protocol_v1.json"))
ORDER = PROTOCOL["order"]
LABELS_34 = ["BenignTraffic"] + [f"Attack{i:02d}" for i in range(1, 34)]
SEEDS = [101, 202, 303, 404, 505]


def matrix(support, error_share, rng):
    """Integer matrix with the given row supports, benign first: most counts on the diagonal,
    the rest spread, attacks mostly towards benign."""
    n, rows = len(support), []
    for i, s in enumerate(support):
        row = [0] * n
        wrong = rng.randint(0, int(s * error_share))
        row[i] = s - wrong
        for _ in range(wrong):
            row[rng.randrange(n) if i == 0 else rng.choice([0, 0, rng.randrange(n)])] += 1
        rows.append(row)
    return rows


def dataset(labels, seeds, supplementary=()):
    return {"name": "synthetic", "benign": labels[0], "labels": labels,
            "pool": ["lgbm", "xgb", "rf"], "supplementary": list(supplementary), "seeds": seeds,
            "parts": {"selection": "validation", "control": "test"}, "records": []}


def synthetic_34(seed=7):
    """34 classes, 3 models, 5 seeds; the supports are the same for every model, as on one split."""
    rng = random.Random(seed)
    support = {p: [20000] + [rng.randint(20, 3000) for _ in range(33)]
               for p in ("validation", "test")}
    share = {"lgbm": 0.02, "xgb": 0.03, "rf": 0.04}
    return {(m, s, p): matrix(support[p], share[m], rng)
            for m, s, p in product(share, SEEDS, support)}


def test_risk03_is_reproduced_unchanged_by_the_generic_path(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert [name for name, ok in risk03_checks().items() if not ok] == []


def test_34_classes_in_the_nine_scenarios():
    matrices, ds = synthetic_34(), dataset(LABELS_34, SEEDS)
    for pi, lam in product(PROTOCOL["scenarios"]["pi"], PROTOCOL["scenarios"]["lambda"]):
        r = scenario(ds, matrices, Fraction(pi), Fraction(lam), PROTOCOL)
        assert len(r["q"]) == 33 and sum(r["q"]) == 1
        assert r["minimax"][0] == r["choice"]["nominal"]  # delta = 0
        assert sorted(r["minimax"]) == [0, Fraction(1, 4), Fraction(1, 2), 1]
        for rows in r["frontier"].values():  # exact partition of [0, 1]
            assert (rows[0]["lo"], rows[-1]["hi"]) == (0, 1)
            assert all(x["hi"] == y["lo"] for x, y in zip(rows, rows[1:]))


def test_a_dominant_supplementary_model_leaves_the_main_pool_unchanged():
    matrices, ds = synthetic_34(), dataset(LABELS_34, SEEDS, supplementary=["logreg"])
    perfect = {("logreg", s, p): [[sum(row) * (i == j) for j in range(34)]
                                  for i, row in enumerate(matrices["lgbm", s, p])]
               for s, p in product(SEEDS, ("validation", "test"))}  # no error, Macro-F1 = 1
    for pi, lam in product(PROTOCOL["scenarios"]["pi"], PROTOCOL["scenarios"]["lambda"]):
        pi, lam = Fraction(pi), Fraction(lam)
        main = scenario(ds, matrices, pi, lam, PROTOCOL)
        assert scenario(ds, {**matrices, **perfect}, pi, lam, PROTOCOL) == main
        wider = scenario(ds, {**matrices, **perfect}, pi, lam, PROTOCOL,
                         ds["pool"] + ds["supplementary"])
        assert {*wider["choice"].values(), *wider["minimax"].values()} == {"logreg"}
    del matrices["rf", 303, "test"]
    with pytest.raises(InputError, match="missing records of the pool"):
        scenario(ds, {**matrices, **perfect}, pi, lam, PROTOCOL)


def test_every_entry_must_be_a_non_negative_integer_count():
    labels, good = ["Benign", "Scan", "Flood"], [[50, 1, 2], [3, 20, 0], [1, 0, 30]]
    assert align(labels, good, labels) == good
    for bad in (-1, True, False, 2.5, 4.0, float("nan"), float("inf"), None, "3"):
        mat = [row[:] for row in good]
        mat[1][2] = bad
        with pytest.raises(InputError, match=r"entry \[Scan, Flood\] = .* is not a non-negative"):
            align(labels, mat, labels)


def test_label_order_is_matched_by_name_and_wrong_labels_stop():
    matrices, rng, shuffled = synthetic_34(), random.Random(1), {}
    for key, mat in matrices.items():
        order = rng.sample(LABELS_34, len(LABELS_34))
        at = [LABELS_34.index(c) for c in order]
        shuffled[key] = align(order, [[mat[i][j] for j in at] for i in at], LABELS_34)
    assert shuffled == matrices
    mat = matrices["lgbm", 101, "validation"]
    for labels, m in [(LABELS_34[:-1] + [LABELS_34[0]], mat),  # duplicate
                      (LABELS_34[:-1] + ["Other"], mat),        # different name
                      (LABELS_34, mat[:-1])]:                   # not square
        with pytest.raises(InputError):
            align(labels, m, LABELS_34)


def test_zero_or_different_supports_stop_the_analysis():
    matrices, ds = synthetic_34(), dataset(LABELS_34, SEEDS)
    zero = [[0] * 34 if i == 5 else row for i, row in enumerate(matrices["xgb", 303, "test"])]
    with pytest.raises(InputError, match=r"zero support for \['Attack05'\]"):
        scenario(ds, {**matrices, ("xgb", 303, "test"): zero}, Fraction(1, 100), 10, PROTOCOL)
    other = [row[:] for row in matrices["xgb", 303, "validation"]]
    other[5][5] += 1
    other = {**matrices, ("xgb", 303, "validation"): other}
    with pytest.raises(InputError, match="q is not unique"):
        scenario(ds, other, Fraction(1, 100), 10, PROTOCOL)


def test_ties_follow_nominal_cost_then_macro_f1_then_the_fixed_order():
    deltas = [Fraction(0), Fraction(1, 2), Fraction(1)]
    chosen = lambda r: {*r["choice"].values(), *r["minimax"].values()}
    same = {"validation": {m: [3, 4] for m in ("rf", "xgb", "lgbm")}}
    assert chosen(decide([1, 1], same, {"lgbm": .9, "xgb": .9, "rf": .9}, ORDER, deltas)) \
        == {"lgbm"}
    assert chosen(decide([1, 1], same, {"lgbm": .9, "xgb": .9, "rf": .95}, ORDER, deltas)) \
        == {"rf"}
    cheaper = {"validation": {"lgbm": [5, 5], "xgb": [4, 4], "rf": [4, 4]}}
    r = decide([1, 1], cheaper, {"lgbm": .9, "xgb": .9, "rf": .9}, ORDER, deltas)
    assert r["choice"]["macro_f1"] == "xgb"  # equal Macro-F1: lower cost, then the order
    # same nominal cost with opposite errors: radius 0, same regret, Macro-F1 decides
    parity = {"validation": {"lgbm": [2, 0], "xgb": [0, 2], "rf": [9, 9]}}
    r = decide([1, 1], parity, {"lgbm": .9, "xgb": .95, "rf": .99}, ORDER, deltas)
    pair = next(x for x in r["pairwise"]["validation"] if {x["m"], x["k"]} == {"lgbm", "xgb"})
    assert (pair["a"], pair["b"], pair["radius"]) == (0, 4, 0)
    assert r["choice"] == {"macro_f1": "rf", "nominal": "xgb"}
    assert set(r["minimax"].values()) == {"xgb"}


def test_class_conditional_normalisation():
    labels, pi, lam = ["Benign", "A", "B"], Fraction(1, 100), Fraction(10)
    w0 = [lam, lam, 1, 1]
    q = [Fraction(3, 10), Fraction(7, 10)]  # validation supports 30 and 70
    rates = lambda mat: rate_vector(errors(mat, labels, "Benign"), q, pi)
    some = [[880, 15, 5], [3, 25, 2], [10, 1, 59]]
    assert rates([[c * 7 for c in row] for row in some]) == rates(some)  # size does not matter
    assert cost(w0, rates([[900, 0, 0], [0, 30, 0], [0, 0, 70]])) == 0    # no error
    blind = rates([[900, 0, 0], [30, 0, 0], [70, 0, 0]])                   # every attack missed
    assert sum(blind[:2]) == pi and blind[2:] == [0, 0] and cost(w0, blind) == lam * pi
    assert rates([[0, 900, 0], [0, 30, 0], [0, 0, 70]])[2:] == [1 - pi, 0]  # benign all to A
    # another class mix on test with the same class-conditional rates: same cost, q stays fixed
    assert cost(w0, rates([[880, 15, 5], [30, 250, 20], [10, 1, 59]])) == cost(w0, rates(some))


def test_read_inputs_checks_hash_labels_and_supports(tmp_path):
    labels, rng = ["Benign", "Scan", "Flood", "Exfil"], random.Random(5)
    ds, matrices = dataset(labels, [1], supplementary=["kan"]), {}

    def write(name, names, mat):
        path = str(tmp_path / name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"labels": names, "matrix": mat}, f)
        return path

    for m, p in product(ds["pool"], ("validation", "test")):
        matrices[m, 1, p] = mat = matrix([500, 40, 60, 30], 0.1, rng)
        stored = [row[::-1] for row in mat[::-1]]  # labels saved in reverse order
        path = write(f"{m}_{p}.json", labels[::-1], stored)
        ds["records"].append({"model": m, "seed": 1, "part": p, "path": path,
                              "sha256": sha256(path)})
    found, issues, missing = read_inputs(ds)
    assert (found, issues) == (matrices, [])
    assert missing == {"required": [], "optional": [("kan", 1, "validation"), ("kan", 1, "test")]}
    scenario(ds, found, Fraction(1, 100), 10, PROTOCOL)  # no KAN: the main comparison still runs

    record = {(e["model"], e["part"]): e for e in ds["records"]}
    nan = [row[:] for row in matrices["rf", 1, "validation"]]
    nan[2][1] = float("nan")
    record["rf", "validation"]["sha256"] = sha256(write("rf_validation.json", labels, nan))
    changed = [row[:] for row in matrices["rf", 1, "test"]]
    changed[1][1] += 1
    write("rf_test.json", labels, changed)  # hash left as pinned
    record["xgb", "test"]["sha256"] = sha256(write("xgb_test.json", labels, changed))
    renamed = write("lgbm_validation.json", labels[:3] + ["Exfil_"],
                    matrices["lgbm", 1, "validation"])
    record["lgbm", "validation"]["sha256"] = sha256(renamed)
    found, issues, _ = read_inputs(ds)
    assert ("rf", 1, "validation") not in found
    issues = "\n".join(issues)
    assert ("synthetic ('rf', 1, 'validation'): entry [Flood, Scan] = nan is not a non-negative"
            " integer count") in issues
    assert "('rf', 1, 'test'): sha256 differs from the contract" in issues
    assert "seed 1 test: supports differ between models" in issues
    assert "('lgbm', 1, 'validation'): labels differ from the contract" in issues


def test_declared_contract(monkeypatch):
    monkeypatch.chdir(ROOT)
    datasets = load(PROTOCOL["inputs"])["datasets"]
    found, issues, missing = read_inputs(datasets["xiiotid"])
    assert (len(found), issues, missing) == (8, [], {"required": [], "optional": []})
    found, issues, missing = read_inputs(datasets["ciciot2023"])
    assert (found, issues) == ({}, [])
    assert {k: {m for m, _, _ in v} for k, v in missing.items()} == {
        "required": {"lgbm", "xgb", "rf"}, "optional": {"kan"}}
    assert (len(missing["required"]), len(missing["optional"])) == (3 * 5 * 2, 5 * 2)
