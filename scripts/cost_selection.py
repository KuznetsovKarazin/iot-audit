"""Generic IDS choice by Macro-F1, nominal cost and minimax regret, for any set of labels.

Reads the confusion matrices declared in configs/common_inputs_v1.json for the common protocol
v1 (configs/common_protocol_v1.json): labels, benign class, models, parts and seeds come from the
configuration. No training and no inference. Margins, radii and minimax regret use the RISK-03
formula, imported unchanged from scripts/xiiotid_cost_robustness.py.

    python scripts/cost_selection.py                    # declared inputs: hashes, labels, supports
    python scripts/cost_selection.py --risk03           # RISK-03 through the generic path
    python scripts/cost_selection.py --dataset xiiotid  # tables and manifest in reports_common/
"""
from __future__ import annotations
import argparse, json, os, platform, sys
from fractions import Fraction
from itertools import product

import xiiotid_cost_robustness as risk03
from xiiotid_cost_regions import sha256, write_csv
from xiiotid_cost_robustness import (cost, git_output, max_regret, merged, pairwise,
                                     regret_intervals, regret_lines)


class InputError(ValueError):
    """An input outside the contract: reported, never repaired."""


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def align(labels, matrix, order):
    """Rows and columns in the order of `order`, matched by name, never by position. Every
    entry must be a non-negative int: booleans, floats, non-finite values and the rest stop."""
    if sorted(labels) != sorted(order):
        raise InputError(f"labels differ from the contract: {labels}")
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise InputError("the matrix is not square in the number of labels")
    bad = [(labels[i], labels[j], x) for i, row in enumerate(matrix) for j, x in enumerate(row)
           if type(x) is not int or x < 0]
    if bad:
        i, j, x = bad[0]
        more = f" ({len(bad) - 1} more)" if len(bad) > 1 else ""
        raise InputError(f"entry [{i}, {j}] = {x!r} is not a non-negative integer count{more}")
    at = [labels.index(c) for c in order]
    return [[matrix[i][j] for j in at] for i in at]


def errors(matrix, labels, benign):
    """Per attack class, in label order: missed (attack -> benign), false alarms
    (benign -> attack), confusions with other attacks (no cost) and supports.
    A zero support stops the analysis."""
    b = labels.index(benign)
    attacks = [i for i in range(len(labels)) if i != b]
    err = {"classes": [labels[i] for i in attacks],
           "missed": [matrix[i][b] for i in attacks],
           "false_alarms": [matrix[b][i] for i in attacks],
           "attack_to_attack": [sum(matrix[i][j] for j in attacks if j != i) for i in attacks],
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
    lines = {p: {m: regret_lines(w0, v, m) for m in pool} for p, v in vectors.items()}
    regret = {d: {p: {m: max_regret(lines[p][m], d) for m in pool} for p in vectors}
              for d in deltas}
    choice = {"macro_f1": min(pool, key=lambda m: (-f1[m], nominal[selection][m], pool.index(m))),
              "nominal": min(pool, key=keys.get)}
    return {"nominal": nominal, "choice": choice, "keys": keys, "regret": regret,
            "minimax": {d: min(pool, key=lambda m: (regret[d][selection][m], keys[m]))
                        for d in deltas},
            "pairwise": {p: pairwise(w0, v, nominal[p], pool) for p, v in vectors.items()},
            "frontier": {p: regret_intervals(w0, v, keys) for p, v in vectors.items()}}


def attack_mix(ds, matrices):
    """q_i from the selection supports, which must be the same in every record."""
    sel = ds["parts"]["selection"]
    mixes = {tuple(errors(mat, ds["labels"], ds["benign"])["support"])
             for (_, _, p), mat in matrices.items() if p == sel}
    if len(mixes) != 1:
        raise InputError(f"{sel} supports differ between records, so q is not unique")
    support = mixes.pop()
    return [Fraction(s, sum(support)) for s in support]


def scenario(ds, matrices, pi, lam, protocol, pool=None):
    """One (pi, lambda) scenario of protocol v1: the choice uses the mean over seeds, never the
    best seed; each single seed is also decided apart, for the report only. Only the models of
    `pool`, by default the main pool of the contract, enter q, the choices, the margins, the
    frontiers and the regret; every other matrix is left out."""
    labels, benign, seeds = ds["labels"], ds["benign"], ds["seeds"]
    sel, parts, pool = ds["parts"]["selection"], list(ds["parts"].values()), pool or ds["pool"]
    absent = [(m, s, p) for m in pool for s in seeds for p in parts if (m, s, p) not in matrices]
    if absent:
        raise InputError(f"missing records of the pool: {absent}")
    matrices = {key: mat for key, mat in matrices.items() if key[0] in pool}
    q = attack_mix(ds, matrices)
    vec = {(m, s, p): rate_vector(errors(matrices[m, s, p], labels, benign), q, pi)
           for m in pool for s in seeds for p in parts}
    f1 = {(m, s): macro_f1(matrices[m, s, sel]) for m in pool for s in seeds}
    mean = lambda xs: sum(xs, Fraction(0)) / len(xs)
    w0 = [lam] * len(q) + [Fraction(1)] * len(q)
    deltas = [Fraction(d) for d in protocol["robustness"]["delta_summary"]]
    on = lambda ss: decide(
        w0, {p: {m: [mean(x) for x in zip(*(vec[m, s, p] for s in ss))] for m in pool}
             for p in parts}, {m: mean([f1[m, s] for s in ss]) for m in pool},
        protocol["order"], deltas, sel)
    return {"q": q, **on(seeds), "seeds": {s: on([s]) for s in seeds}}


def run(ds, matrices, protocol):
    """The tables of protocol v1 for one dataset, choices frozen on the selection part."""
    name, sel, ctl = ds["name"], ds["parts"]["selection"], ds["parts"]["control"]
    dec = lambda x: format(float(x), ".10g")
    per = lambda x: format(float(x * 100000), ".10g")
    q = attack_mix(ds, {k: mat for k, mat in matrices.items() if k[0] in ds["pool"]})
    counts = []
    for (m, s, p), mat in matrices.items():
        e = errors(mat, ds["labels"], ds["benign"])
        for i, c in enumerate(e["classes"]):
            counts.append([name, p, m, s, c, e["support"][i], e["missed"][i],
                           dec(Fraction(e["missed"][i], e["support"][i])), e["false_alarms"][i],
                           dec(Fraction(e["false_alarms"][i], e["benign_support"])),
                           e["attack_to_attack"][i], dec(q[i])])
    extra = [m for m in ds["supplementary"] if all((m, s, p) in matrices for s in ds["seeds"]
                                                   for p in (sel, ctl))]
    pools = {"main": ds["pool"], **({"supplementary": ds["pool"] + extra} if extra else {})}
    choices, frontiers, radii = [], [], []
    for (pi, lam), (pool_name, pool) in product(
            product(protocol["scenarios"]["pi"], protocol["scenarios"]["lambda"]), pools.items()):
        r = scenario(ds, matrices, Fraction(pi), Fraction(lam), protocol, pool)
        tag = [name, pool_name, pi, lam]
        for d, minimax in r["minimax"].items():
            picks = {**r["choice"], "minimax_regret": minimax}
            runs = [("mean", r), *r["seeds"].items()]
            for (policy, m), (seed, x) in product(picks.items(), runs):
                regret = x["regret"][d][ctl]
                best = min(regret, key=lambda k: (regret[k], x["keys"][k]))  # retrospective only
                choices.append([*tag, d, policy, m, len(set(picks.values())) == 1, seed,
                                per(x["nominal"][sel][m]), per(x["regret"][d][sel][m]),
                                per(x["nominal"][ctl][m]), per(regret[m]), best, per(regret[best])])
        for p in (sel, ctl):
            for row in merged(r["frontier"][p]):
                frontiers.append([*tag, p, row["lo"], row["hi"], row["lo_closed"], row["hi_closed"],
                                  dec(row["lo"]), dec(row["hi"]), row["selected"]])
            for row in r["pairwise"][p]:
                none = row["radius"] is None
                radii.append([*tag, p, row["m"], row["k"], per(-row["a"]),
                              "" if none else row["radius"], "" if none else dec(row["radius"])])
    scenario_cols = ["dataset", "pool", "pi", "lambda"]
    return {
        "normalized_counts.csv": (
            ["dataset", "part", "model", "seed", "class", "support", "missed", "missed_rate",
             "false_alarms", "false_alarm_rate", "attack_to_attack", "q"], counts),
        "choices.csv": (
            [*scenario_cols, "delta", "policy", "model", "all_policies_agree", "seed",
             f"{sel}_cost_per_100k", f"{sel}_max_regret_per_100k", f"{ctl}_cost_per_100k",
             f"{ctl}_max_regret_per_100k", f"{ctl}_best_retrospective",
             f"{ctl}_best_max_regret_per_100k"], choices),
        "frontiers.csv": (
            [*scenario_cols, "part", "delta_from", "delta_to", "from_closed", "to_closed",
             "delta_from_decimal", "delta_to_decimal", "min_regret_model"], frontiers),
        "radii.csv": (
            [*scenario_cols, "part", "model_m", "model_k", "margin_per_100k", "radius",
             "radius_decimal"], radii),
    }


def write_outputs(tables, ds, protocol_path, protocol, outdir):
    """The tables, then a manifest with the hashes of inputs, protocol, code and outputs."""
    os.makedirs(outdir, exist_ok=True)
    for f, (header, rows) in tables.items():
        write_csv(os.path.join(outdir, f), rows, header)
    code = ["scripts/cost_selection.py", "scripts/xiiotid_cost_robustness.py",
            "scripts/xiiotid_cost_regions.py"]
    inputs = sorted({e["path"] for e in ds["records"]})
    manifest = {
        "dataset": ds["name"],
        "base_commit": protocol["base_commit"],
        "analysis_commit": git_output("rev-parse", "HEAD"),
        "analysis_files_clean": not git_output("status", "--porcelain", "--", protocol_path,
                                               protocol["inputs"], *code, *inputs),
        "python": platform.python_version(),
        "new_fits": 0,
        "new_inferences": 0,
        "protocol_sha256": {p: sha256(p) for p in (protocol_path, protocol["inputs"])},
        "inputs_sha256": {p: sha256(p) for p in inputs},
        "code_sha256": {p: sha256(p) for p in code},
        "outputs_sha256": {f: sha256(os.path.join(outdir, f)) for f in tables},
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def read_inputs(ds):
    """The declared matrices of one dataset, checked against the contract: pinned sha256,
    labels matched by name, integer counts, no zero support, same supports for every model of a
    part and seed. Returns the matrices, the issues and the records not yet available: required
    for the main pool, optional for the supplementary models, whose absence blocks nothing."""
    parts = list(ds["parts"].values())
    expected = {kind: [(m, s, p) for m in ds[models] for s in ds["seeds"] for p in parts]
                for kind, models in (("required", "pool"), ("optional", "supplementary"))}
    declared = {(e["model"], e["seed"], e["part"]): e for e in ds["records"]}
    absent = lambda k: k not in declared or not os.path.exists(declared[k]["path"])
    missing = {kind: [k for k in keys if absent(k)] for kind, keys in expected.items()}
    matrices, issues = {}, []
    for key in [k for keys in expected.values() for k in keys if not absent(k)]:
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
    ap.add_argument("--dataset", help="run protocol v1 on one dataset of the input contract")
    ap.add_argument("--outdir", default="reports_common")
    args = ap.parse_args()
    if args.risk03:
        results = risk03_checks()
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'ERROR'} [risk03] {name}")
        return int(not all(results.values()))
    protocol = load(args.protocol)
    datasets = load(protocol["inputs"])["datasets"]
    if args.dataset:
        ds = datasets[args.dataset]
        matrices, issues, missing = read_inputs(ds)
        if issues or missing["required"]:
            print(f"ERROR {ds['name']}: inputs incomplete or outside the contract, "
                  f"nothing computed")
            return 1
        outdir = os.path.join(args.outdir, args.dataset)
        write_outputs(run(ds, matrices, protocol), ds, args.protocol, protocol, outdir)
        print(f"{ds['name']}: tables and manifest written to {outdir}/")
        return 0
    status = 0
    for ds in datasets.values():
        matrices, issues, missing = read_inputs(ds)
        print(f"{ds['name']}: {len(matrices)} records read, {len(issues)} issues, "
              f"{len(missing['required'])} required missing (main pool), "
              f"{len(missing['optional'])} optional missing (supplementary)")
        for issue in issues:
            print(f"ERROR {issue}")
        for kind, keys in missing.items():
            for m in dict.fromkeys(k[0] for k in keys):
                seeds = ", ".join(f"{p} {s}" for m2, s, p in keys if m2 == m)
                print(f"MISSING {kind} {ds['name']} {m}: {seeds}")
        status |= bool(issues)
    return status


if __name__ == "__main__":
    sys.exit(main())
