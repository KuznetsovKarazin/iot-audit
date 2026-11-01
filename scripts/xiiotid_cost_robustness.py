"""Robustness of the X-IIoTID model choice to independent changes of the per-class costs.

No training and no dataset: exact rational arithmetic on the validation and test confusion
matrices already accepted in reports_xiiot/, under the protocol fixed in
configs/xiiotid_cost_robustness.json. Writes the CSV tables, the figure and the manifest into
reports_risk03/ and exits with status 1 if an input hash or a check fails.

    python scripts/xiiotid_cost_robustness.py
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from fractions import Fraction
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xiiotid_cost_regions import PARTS, SERIES_COLOURS, sha256, summarise, write_csv

ROLES = {"validation": "selection", "test": "retrospective_diagnostic"}
METRICS = "reports_xiiot/metrics_{}.json"


def component_names(labels):
    attacks = [c for c in labels if c != "Normal"]
    return [f"fn:{c}" for c in attacks] + [f"fp:{c}" for c in attacks]


def weight_vector(labels, costs):
    """w0 in component order: cost_fn then cost_fp of every attack class."""
    attacks = [c for c in labels if c != "Normal"]
    return [costs[c]["cost_fn"] for c in attacks] + [costs[c]["cost_fp"] for c in attacks]


def error_vector(matrix, labels):
    """The 18 counts: attacks classified as Normal, then Normal classified as each attack."""
    n = labels.index("Normal")
    attacks = [i for i in range(len(labels)) if i != n]
    return [matrix[i][n] for i in attacks] + [matrix[n][j] for j in attacks]


def cost(w, e):
    return sum(wi * ei for wi, ei in zip(w, e))


def terms(w0, e_m, e_k):
    """Worst case of m against k is a + delta * b, with a = w0.d and b = sum w0_i |d_i|."""
    d = [x - y for x, y in zip(e_m, e_k)]
    return sum(w * di for w, di in zip(w0, d)), sum(w * abs(di) for w, di in zip(w0, d))


def radius(a, b):
    """Largest delta with m not costlier than k; None for equal cost (b = 0) or a > 0."""
    return Fraction(-a, b) if b and a <= 0 else None


def extreme_vector(w0, e_m, e_k, delta):
    """The weights inside the box that are worst for m against k."""
    return [w * (1 + delta * ((x > y) - (x < y))) for w, x, y in zip(w0, e_m, e_k)]


def regret_lines(w0, vectors, m):
    return [terms(w0, vectors[m], vectors[k]) for k in vectors if k != m]


def max_regret(lines, delta):
    return max(0, *(a + delta * b for a, b in lines))


def breakpoints(lines, lo=Fraction(0), hi=Fraction(1)):
    """Every delta where two regret lines, or one line and zero, can cross."""
    points = {lo, hi}
    for (a0, b0), (a1, b1) in combinations(sorted({(0, 0), *lines}), 2):
        if b0 != b1 and lo <= (p := Fraction(a1 - a0, b0 - b1)) <= hi:
            points.add(p)
    return sorted(points)


def tie_keys(nominal, macro_f1, pool):
    """Protocol tie-break: lowest nominal cost, highest Macro-F1, then the fixed order."""
    return {m: (nominal[m], -macro_f1[m], pool.index(m)) for m in pool}


def regret_intervals(w0, vectors, keys):
    """Partition of delta in [0, 1] where the selected model and every R_m slope are constant.

    Points and the open pieces between them are visited in order and merged while nothing
    changes, so a tie resolved differently at a single delta gets its own degenerate interval.
    """
    lines = {m: regret_lines(w0, vectors, m) for m in vectors}
    regret = lambda delta: {m: max_regret(lines[m], delta) for m in vectors}
    choose = lambda delta: min(keys, key=lambda m: (regret(delta)[m], keys[m]))
    points = breakpoints([line for ls in lines.values() for line in ls])
    rows, now = [], None
    for i, p in enumerate(points):
        selected = choose(p)
        if now and now["selected"] == selected:
            now.update(hi=p, hi_closed=True)
        else:
            now = dict(lo=p, lo_closed=True, hi=p, hi_closed=True, selected=selected, slopes=None)
            rows.append(now)
        if i + 1 == len(points):
            break
        q = points[i + 1]
        selected = choose((p + q) / 2)
        slopes = tuple((regret(q)[m] - regret(p)[m]) / (q - p) for m in vectors)
        if now["selected"] == selected and now["slopes"] in (None, slopes):
            now.update(hi=q, hi_closed=False, slopes=slopes)
        else:
            now = dict(lo=p, lo_closed=False, hi=q, hi_closed=False, selected=selected, slopes=slopes)
            rows.append(now)
    for row in rows:
        row["regret_lo"], row["regret_hi"] = regret(row["lo"]), regret(row["hi"])
    return rows


def pairwise(w0, vectors, nominal, pool):
    """Each pair oriented with m the nominally cheaper model, ties in the fixed order."""
    rows = []
    for x, y in combinations(pool, 2):
        m, k = sorted((x, y), key=lambda v: (nominal[v], pool.index(v)))
        a, b = terms(w0, vectors[m], vectors[k])
        r = radius(a, b)
        extreme = extreme_vector(w0, vectors[m], vectors[k], r) if r is not None else None
        rows.append(dict(m=m, k=k, a=a, b=b, radius=r, extreme=extreme))
    return rows


def analyse(metrics, costs, protocol):
    """All the tables; validation decides, and the test part only inherits its tie-break."""
    pool = protocol["pool"]
    labels = protocol["labels"]
    w0 = weight_vector(labels, costs)
    macro_f1 = {m: metrics[m]["macro_f1_validation"] for m in pool}
    vectors, nominal = {}, {}
    for part, key in PARTS.items():
        vectors[part] = {m: error_vector(metrics[m][key]["matrix"], labels) for m in pool}
        nominal[part] = {m: cost(w0, vectors[part][m]) for m in pool}
    keys = tie_keys(nominal["validation"], macro_f1, pool)
    return {
        "w0": w0, "labels": labels, "pool": pool, "vectors": vectors, "nominal": nominal,
        "nominal_choice": min(pool, key=keys.get),
        "macro_f1_choice": max(pool, key=lambda m: (macro_f1[m], -pool.index(m))),
        "pairwise": {part: pairwise(w0, vectors[part], nominal[part], pool) for part in PARTS},
        "regret": {part: regret_intervals(w0, vectors[part], keys) for part in PARTS},
    }


def decimal(x):
    return round(float(x), 6)


def write_tables(result, metrics, costs, outdir):
    pool, labels, names = result["pool"], result["labels"], component_names(result["labels"])
    write_csv(os.path.join(outdir, "counts.csv"),
              [[part, ROLES[part], m, result["nominal"][part][m], *result["vectors"][part][m],
                summarise(metrics[m][key]["matrix"], labels, costs)["attack_confusions_unweighted"]]
               for part, key in PARTS.items() for m in pool],
              ["part", "role", "model", "nominal_cost", *names, "attack_to_attack_unweighted"])
    write_csv(os.path.join(outdir, "pairwise.csv"),
              [[part, ROLES[part], r["m"], r["k"], result["nominal"][part][r["m"]],
                result["nominal"][part][r["k"]], -r["a"], r["b"],
                "" if r["radius"] is None else r["radius"],
                "" if r["radius"] is None else decimal(r["radius"]),
                "" if r["extreme"] is None else cost(r["extreme"], result["vectors"][part][r["m"]]),
                *(r["extreme"] or [""] * len(names))]
               for part in PARTS for r in result["pairwise"][part]],
              ["part", "role", "model_m", "model_k", "cost_m", "cost_k", "margin", "denominator",
               "radius", "radius_decimal", "cost_of_both_at_extreme", *[f"w_extreme:{n}" for n in names]])
    write_csv(os.path.join(outdir, "minimax_regret.csv"),
              [[part, ROLES[part], r["lo"], r["hi"], r["lo_closed"], r["hi_closed"], decimal(r["lo"]),
                decimal(r["hi"]), r["selected"], result["nominal_choice"], result["macro_f1_choice"],
                *[v for m in pool for v in (r["regret_lo"][m], r["regret_hi"][m])]]
               for part in PARTS for r in result["regret"][part]],
              ["part", "role", "delta_from", "delta_to", "from_closed", "to_closed", "delta_from_decimal",
               "delta_to_decimal", "min_regret_model", "nominal_choice_on_validation",
               "macro_f1_choice_on_validation",
               *[f"R_{m}_{end}" for m in pool for end in ("from", "to")]])


def figure(result, path):
    pool, rows = result["pool"], result["regret"]["validation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = ((axes[0], pool, "Tutti i modelli"), (axes[1], [m for m in pool if m != "logreg"],
                                                   "Senza logreg, scala più stretta"))
    frontier = min(r["radius"] for r in result["pairwise"]["validation"]
                   if r["m"] == result["nominal_choice"] and r["radius"] is not None)
    for ax, models, title in panels:
        for row in rows:
            ax.axvspan(float(row["lo"]), float(row["hi"]), color="#f1f3f4", linewidth=0, zorder=0)
        for m in models:
            xs = [float(rows[0]["lo"])] + [float(r["hi"]) for r in rows]
            ys = [float(rows[0]["regret_lo"][m])] + [float(r["regret_hi"][m]) for r in rows]
            ax.plot(xs, ys, color=SERIES_COLOURS[m], linewidth=2, label=m, zorder=3)
            ax.annotate(m, (1, ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8,
                        va="center", color="#3c4043")
        ax.axvline(float(frontier), color="#5f6368", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("δ, variazione massima di ciascun peso")
        ax.set_xlim(0, 1.09)
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#e8eaed", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].annotate(f"primo confine δ={float(frontier):.4f}", (float(frontier), axes[1].get_ylim()[1]),
                     xytext=(-4, -10), textcoords="offset points", fontsize=8, ha="right", color="#5f6368")
    axes[0].set_ylabel("Regret massimo R(δ) in unità relative")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    choices = ", ".join(f"{r['selected']} su {'[' if r['lo_closed'] else '('}{float(r['lo']):g}, "
                        f"{float(r['hi']):g}{']' if r['hi_closed'] else ')'}" for r in merged(rows))
    fig.suptitle(f"Validation — scelta minimax regret: {choices}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def merged(rows):
    """Selection intervals only, ignoring the slope changes that split the CSV rows."""
    out = []
    for row in rows:
        if out and out[-1]["selected"] == row["selected"]:
            out[-1] = {**out[-1], "hi": row["hi"], "hi_closed": row["hi_closed"]}
        else:
            out.append(row)
    return out


def git_output(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def write_manifest(protocol_path, script_path, outdir, protocol):
    outputs = sorted(f for f in os.listdir(outdir) if f != "manifest.json")
    tracked = [protocol_path, script_path, "tests/test_cost_robustness.py", *protocol["inputs_sha256"]]
    manifest = {
        "base_commit": protocol["base_commit"],
        "analysis_commit": git_output("rev-parse", "HEAD"),
        "analysis_files_clean": not git_output("status", "--porcelain", "--", *tracked),
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "new_fits": 0,
        "new_inferences": 0,
        "inputs_sha256": {p: sha256(p) for p in protocol["inputs_sha256"]},
        "protocol_sha256": {protocol_path: sha256(protocol_path)},
        "code_sha256": {p: sha256(p) for p in (script_path, "tests/test_cost_robustness.py")},
        "outputs_sha256": {f: sha256(os.path.join(outdir, f)) for f in outputs},
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/xiiotid_cost_robustness.json")
    ap.add_argument("--outdir", default="reports_risk03")
    args = ap.parse_args()

    protocol = json.load(open(args.protocol, encoding="utf-8"))
    wrong = [p for p, h in protocol["inputs_sha256"].items() if sha256(p) != h]
    if wrong:
        print(f"ERROR [risk03] input hash differs from the protocol: {wrong}")
        return 1
    costs = json.load(open(protocol["weights"], encoding="utf-8"))["classes"]
    metrics = {m: json.load(open(METRICS.format(m), encoding="utf-8")) for m in protocol["pool"]}
    os.makedirs(args.outdir, exist_ok=True)

    result = analyse(metrics, costs, protocol)
    write_tables(result, metrics, costs, args.outdir)
    figure(result, os.path.join(args.outdir, "regret_frontier.png"))
    write_manifest(args.protocol, "scripts/xiiotid_cost_robustness.py", args.outdir, protocol)

    labels = protocol["labels"]
    checks = {
        "labels and their order match the protocol":
            all(m[key]["labels"] == labels for m in metrics.values()
                for key in ("confusion_matrix", "confusion_matrix_validation")),
        "the 18 components give the costs of the RISK-02 summary":
            all(result["nominal"][part][v] == s["missed_attack_cost"] + s["false_alarm_cost"]
                for part, key in PARTS.items() for v in protocol["pool"]
                for s in [summarise(metrics[v][key]["matrix"], labels, costs)]),
        "every extreme vector makes the two costs equal":
            all(r["extreme"] is None or cost(r["extreme"], result["vectors"][part][r["m"]])
                == cost(r["extreme"], result["vectors"][part][r["k"]])
                for part in PARTS for r in result["pairwise"][part]),
        "the intervals cover delta in [0, 1] exactly once":
            all(rows[0]["lo"] == 0 and rows[0]["lo_closed"] and rows[-1]["hi"] == 1 and rows[-1]["hi_closed"]
                and all(x["hi"] == y["lo"] and x["hi_closed"] != y["lo_closed"]
                        for x, y in zip(rows, rows[1:]))
                for rows in result["regret"].values()),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'ERROR'} [risk03] {name}")
    print(f"     tables, figure and manifest written to {args.outdir}/")
    return int(not all(checks.values()))


if __name__ == "__main__":
    sys.exit(main())
