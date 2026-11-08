"""Generic IDS choice by Macro-F1, nominal cost and minimax regret, for any set of labels.

Reads the confusion matrices declared in configs/common_inputs_v1.json for the common protocol
v1 (configs/common_protocol_v1.json): labels, benign class, models, parts and seeds come from the
configuration. No training and no inference. Margins, radii and minimax regret use the RISK-03
formula, imported unchanged from scripts/xiiotid_cost_robustness.py.

    python scripts/cost_selection.py            # declared inputs: hashes, labels, supports
    python scripts/cost_selection.py --risk03   # RISK-03 through the generic path
"""
from __future__ import annotations
import argparse, json, os, sys
from fractions import Fraction
from itertools import product

import xiiotid_cost_robustness as risk03
from xiiotid_cost_regions import sha256
from xiiotid_cost_robustness import cost, max_regret, pairwise, regret_intervals, regret_lines


class InputError(ValueError):
    """An input outside the contract: reported, never repaired."""


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def align(labels, matrix, order):
    """Rows and columns in the order of `order`, matched by name, never by position."""
    if sorted(labels) != sorted(order):
        raise InputError(f"labels differ from the contract: {labels}")
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise InputError("the matrix is not square in the number of labels")
    at = [labels.index(c) for c in order]
    return [[matrix[i][j] for j in at] for i in at]


def errors(matrix, labels, benign):
    """Per attack class, in label order: missed (attack -> benign), false alarms
    (benign -> attack) and supports. A zero support stops the analysis."""
    b = labels.index(benign)
    attacks = [i for i in range(len(labels)) if i != b]
    err = {"classes": [labels[i] for i in attacks],
           "missed": [matrix[i][b] for i in attacks],
           "false_alarms": [matrix[b][i] for i in attacks],
           "support": [sum(matrix[i]) for i in attacks],
           "benign_support": sum(matrix[b])}
    zero = [labels[i] for i in range(len(labels)) if not sum(matrix[i])]
    if zero:
        raise InputError(f"zero support for {zero}")
    return err


def count_vector(err):
    """RISK-03 coefficients: raw counts of missed attacks, then of false alarms."""
    return err["missed"] + err["false_alarms"]


def rate_vector(err, q, pi):
    """Protocol v1: e_FN,i = pi q_i CM[i, b] / support_i,
    then e_FP,i = (1 - pi) CM[b, i] / support_b."""
    return ([pi * qi * Fraction(x, s) for qi, x, s in zip(q, err["missed"], err["support"])]
            + [(1 - pi) * Fraction(x, err["benign_support"]) for x in err["false_alarms"]])


def macro_f1(matrix):
    """Unweighted mean over every class of 2TP / (2TP + FP + FN), exact."""
    rows, cols, n = [sum(r) for r in matrix], [sum(c) for c in zip(*matrix)], len(matrix)
    return sum(Fraction(2 * matrix[i][i], rows[i] + cols[i]) for i in range(n)) / n


def decide(w0, vectors, f1, order, deltas, selection="validation"):
    """Macro-F1, nominal and minimax-regret choices on the selection part, exact frontiers.

    vectors[part][model] holds the coefficients of the linear cost. Ties: nominal cost, then
    Macro-F1, then `order`. The other part inherits that tie-break and never enters the choice.
    """
    pool = [m for m in order if m in vectors[selection]]
    nominal = {p: {m: cost(w0, v[m]) for m in pool} for p, v in vectors.items()}
    keys = {m: (nominal[selection][m], -f1[m], pool.index(m)) for m in pool}
    lines = {m: regret_lines(w0, vectors[selection], m) for m in pool}
    choice = {"macro_f1": min(pool, key=lambda m: (-f1[m], nominal[selection][m], pool.index(m))),
              "nominal": min(pool, key=keys.get)}
    return {"nominal": nominal, "choice": choice,
            "minimax": {d: min(pool, key=lambda m: (max_regret(lines[m], d), keys[m]))
                        for d in deltas},
            "pairwise": {p: pairwise(w0, v, nominal[p], pool) for p, v in vectors.items()},
            "frontier": {p: regret_intervals(w0, v, keys) for p, v in vectors.items()}}


def scenario(ds, matrices, pi, lam, protocol):
    """One (pi, lambda) scenario of protocol v1, on the mean over seeds, never the best seed."""
    labels, benign, seeds = ds["labels"], ds["benign"], ds["seeds"]
    sel = ds["parts"]["selection"]
    err = {key: errors(mat, labels, benign) for key, mat in matrices.items()}
    mixes = {tuple(e["support"]) for (_, _, p), e in err.items() if p == sel}
    if len(mixes) != 1:
        raise InputError(f"{sel} supports differ between records, so q is not unique")
    support = mixes.pop()
    q = [Fraction(s, sum(support)) for s in support]
    models = sorted({m for m, _, _ in matrices}, key=protocol["order"].index)
    mean = lambda vs: [sum(xs, Fraction(0)) / len(vs) for xs in zip(*vs)]
    vectors = {p: {m: mean([rate_vector(err[m, s, p], q, pi) for s in seeds]) for m in models}
               for p in ds["parts"].values()}
    f1 = {m: sum(macro_f1(matrices[m, s, sel]) for s in seeds) / len(seeds) for m in models}
    w0 = [lam] * len(q) + [Fraction(1)] * len(q)
    deltas = [Fraction(d) for d in protocol["robustness"]["delta_summary"]]
    return {"q": q, **decide(w0, vectors, f1, protocol["order"], deltas, sel)}


def read_inputs(ds):
    """The declared matrices of one dataset, checked against the contract: pinned sha256,
    labels matched by name, no zero support, same supports for every model of a part and seed.
    Returns the matrices, the issues and the expected records not yet available."""
    parts = list(ds["parts"].values())
    expected = [(m, s, p) for m in ds["pool"] + ds["supplementary"] for s in ds["seeds"]
                for p in parts]
    declared = {(e["model"], e["seed"], e["part"]): e for e in ds["records"]}
    missing = [k for k in expected if k not in declared or not os.path.exists(declared[k]["path"])]
    matrices, issues = {}, []
    for key in [k for k in expected if k not in missing]:
        e = declared[key]
        if sha256(e["path"]) != e["sha256"]:
            issues.append(f"{ds['name']} {key}: sha256 differs from the contract")
            continue
        data = load(e["path"])
        record = data[e["key"]] if e.get("key") else data
        try:
            matrix = align(record["labels"], record["matrix"], ds["labels"])
            errors(matrix, ds["labels"], ds["benign"])
            matrices[key] = matrix
        except InputError as exc:
            issues.append(f"{ds['name']} {key}: {exc}")
    for s, p in product(ds["seeds"], parts):
        same = [mat for (_, s2, p2), mat in matrices.items() if (s2, p2) == (s, p)]
        if len({tuple(map(sum, mat)) for mat in same}) > 1:
            issues.append(f"{ds['name']} seed {s} {p}: supports differ between models")
    return matrices, issues, missing


def risk03_checks():
    """RISK-03 through the generic path, against the RISK-03 module, which stays unchanged."""
    protocol = load("configs/xiiotid_cost_robustness.json")
    costs = load(protocol["weights"])["classes"]
    labels, pool = protocol["labels"], protocol["pool"]
    metrics = {m: load(risk03.METRICS.format(m)) for m in pool}
    matrix = lambda m, key: align(metrics[m][key]["labels"], metrics[m][key]["matrix"], labels)
    vectors = {p: {m: count_vector(errors(matrix(m, key), labels, "Normal")) for m in pool}
               for p, key in risk03.PARTS.items()}
    attacks = [c for c in labels if c != "Normal"]
    w0 = [costs[c]["cost_fn"] for c in attacks] + [costs[c]["cost_fp"] for c in attacks]
    new = decide(w0, vectors, {m: metrics[m]["macro_f1_validation"] for m in pool}, pool, [])
    old = risk03.analyse(metrics, costs, protocol)
    return {
        "same 18 coefficients, weights and nominal costs":
            (vectors, w0, new["nominal"]) == (old["vectors"], old["w0"], old["nominal"]),
        "same nominal and Macro-F1 choices":
            new["choice"] == {"nominal": old["nominal_choice"], "macro_f1": old["macro_f1_choice"]},
        "same margins, radii and extreme vectors": new["pairwise"] == old["pairwise"],
        "same minimax-regret intervals on validation and test": new["frontier"] == old["regret"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/common_protocol_v1.json")
    ap.add_argument("--risk03", action="store_true", help="RISK-03 through the generic path")
    args = ap.parse_args()
    if args.risk03:
        results = risk03_checks()
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'ERROR'} [risk03] {name}")
        return int(not all(results.values()))
    status = 0
    for ds in load(load(args.protocol)["inputs"])["datasets"].values():
        matrices, issues, missing = read_inputs(ds)
        print(f"{ds['name']}: {len(matrices)} records read, {len(issues)} issues, "
              f"{len(missing)} missing")
        for issue in issues:
            print(f"ERROR {issue}")
        for m in dict.fromkeys(k[0] for k in missing):
            seeds = ", ".join(f"{p} {s}" for m2, s, p in missing if m2 == m)
            print(f"MISSING {ds['name']} {m}: {seeds}")
        status |= bool(issues)
    return status


if __name__ == "__main__":
    sys.exit(main())
