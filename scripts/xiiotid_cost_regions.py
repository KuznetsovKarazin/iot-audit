"""Cost analysis and choice regions for the X-IIoTID baseline, from the six metrics JSON.

No training and no dataset: the whole analysis is arithmetic on the validation and test
confusion matrices already accepted in reports_xiiot/. Writes the CSV tables and the
choice-regions figure into reports_risk02/ and exits with status 1 if a check fails.

    python scripts/xiiotid_cost_regions.py
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = {"rf": "metrics_rf.json", "xgb": "metrics_xgb.json", "lgbm": "metrics_lgbm.json",
            "logreg": "metrics_logreg.json", "rf_no_host": "metrics_rf_no_host.json",
            "logreg_no_host": "metrics_logreg_no_host.json"}
PARTS = {"validation": "confusion_matrix_validation", "test": "confusion_matrix"}
SERIES_COLOURS = {"lgbm": "#2a78d6", "xgb": "#eb6834", "rf": "#1baf7a", "logreg": "#eda100"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def summarise(matrix, labels, costs):
    """Split the errors into missed attacks, false alarms and attack-to-attack confusions."""
    n = labels.index("Normal")
    k = range(len(labels))
    missed = {labels[i]: matrix[i][n] for i in k if i != n and matrix[i][n]}
    alarms = {labels[j]: matrix[n][j] for j in k if j != n and matrix[n][j]}
    return {
        "missed_attacks": sum(missed.values()),
        "missed_attack_cost": sum(v * costs[c]["cost_fn"] for c, v in missed.items()),
        "false_alarms": sum(alarms.values()),
        "false_alarm_cost": sum(v * costs[c]["cost_fp"] for c, v in alarms.items()),
        "attack_confusions_unweighted": sum(matrix[i][j] for i in k for j in k
                                            if i != n and j != n and i != j),
        "rows": sum(sum(row) for row in matrix),
        "normal_rows": sum(matrix[n]),
    }


def lines_of(metrics, part, costs):
    """The two coefficients of C(lambda) for one variant and one part."""
    summary = summarise(metrics[PARTS[part]]["matrix"], metrics["confusion_matrix"]["labels"], costs)
    return summary["missed_attack_cost"], summary["false_alarm_cost"]


def envelope(lines, pool):
    """Lower envelope of C(lambda) = a + b*lambda on lambda >= 0.

    Returns segments [lambda_from, lambda_to) with the variant preferred inside each one.
    Boundary rule: at a crossing the variants cost the same, and the boundary belongs to the
    segment that starts there, that is to the variant with the smaller false-alarm cost;
    variants equal in both coefficients are ordered by name. The same rule is used by
    select(), so the tables and the regions never disagree. Lines are swept by decreasing
    slope and a line is dropped as soon as it is above the envelope, so equal intercepts and
    several lines crossing at one point are handled.
    """
    stack = []  # (slope, intercept, variant, lambda where it takes over)
    for slope, intercept, v in sorted((-lines[v][1], lines[v][0], v) for v in pool):
        slope = -slope
        if stack and stack[-1][0] == slope:
            continue  # same slope, intercept not lower: never cheaper
        start = 0.0
        while stack:
            slope0, intercept0, _, start0 = stack[-1]
            start = (intercept - intercept0) / (slope0 - slope)
            if start <= start0:
                stack.pop()
                start = 0.0
            else:
                break
        stack.append((slope, intercept, v, start))
    return [(start, stack[i + 1][3] if i + 1 < len(stack) else float("inf"), v)
            for i, (_, _, v, start) in enumerate(stack)]


def select(lines, pool, lam):
    """Selector: cheapest variant at lambda; ties go to the smaller false-alarm cost, then name."""
    return min(pool, key=lambda v: (lines[v][0] + lam * lines[v][1], lines[v][1], v))


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="reports_xiiot")
    ap.add_argument("--outdir", default="reports_risk02")
    ap.add_argument("--costs", default="configs/xiiotid_severity_cost.json")
    ap.add_argument("--lambdas", default="configs/xiiotid_lambda.json")
    args = ap.parse_args()

    costs = json.load(open(args.costs, encoding="utf-8"))["classes"]
    config = json.load(open(args.lambdas, encoding="utf-8"))
    pool, controls = config["selector"]["pool"], config["selector"]["paired_controls"]
    metrics = {v: json.load(open(os.path.join(args.indir, f), encoding="utf-8"))
               for v, f in VARIANTS.items()}
    inputs = {os.path.join(args.indir, f): sha256(os.path.join(args.indir, f))
              for f in VARIANTS.values()}
    inputs.update({p: sha256(p) for p in (args.costs, args.lambdas)})
    os.makedirs(args.outdir, exist_ok=True)

    counts, lines = {}, {}
    for v, m in metrics.items():
        labels = m["confusion_matrix"]["labels"]
        for part, key in PARTS.items():
            s = summarise(m[key]["matrix"], labels, costs)
            counts[(v, part)] = s
            lines[(v, part)] = (s["missed_attack_cost"], s["false_alarm_cost"])

    write_csv(os.path.join(args.outdir, "counts_costs.csv"),
              [[v, part, s["missed_attacks"], s["missed_attack_cost"], s["false_alarms"],
                s["false_alarm_cost"], s["attack_confusions_unweighted"], s["rows"],
                round(s["missed_attack_cost"] / s["rows"] * 1000, 3),
                round(s["false_alarm_cost"] / s["rows"] * 1000, 3),
                round(s["false_alarms"] / s["normal_rows"], 6)]
               for (v, part), s in sorted(counts.items())],
              ["variant", "part", "missed_attacks", "missed_attack_cost", "false_alarms",
               "false_alarm_cost", "attack_confusions_unweighted", "rows",
               "missed_attack_cost_per_1000_rows", "false_alarm_cost_per_1000_rows",
               "false_positive_rate_on_normal"])

    regions = {part: envelope({v: lines[(v, part)] for v in pool}, pool) for part in PARTS}
    write_csv(os.path.join(args.outdir, "choice_regions.csv"),
              [[part, round(lo, 4), "inf" if hi == float("inf") else round(hi, 4), v,
                lines[(v, part)][0], lines[(v, part)][1]]
               for part in PARTS for lo, hi, v in regions[part]],
              ["part", "lambda_from", "lambda_to", "preferred_variant", "C_miss", "C_false_alarm"])

    val = {v: lines[(v, "validation")] for v in pool}
    test = {v: lines[(v, "test")] for v in pool}
    by_f1 = max(pool, key=lambda v: (metrics[v]["macro_f1_validation"], v))
    selector_rows = []
    for lam in config["lambda_grid"]:
        chosen = select(val, pool, lam)
        best_on_test = select(test, pool, lam)  # diagnostic reference only
        selector_rows.append([lam, chosen, round(val[chosen][0] + lam * val[chosen][1], 2),
                              round(test[chosen][0] + lam * test[chosen][1], 2), by_f1,
                              round(test[by_f1][0] + lam * test[by_f1][1], 2), best_on_test,
                              round(test[best_on_test][0] + lam * test[best_on_test][1], 2)])
    write_csv(os.path.join(args.outdir, "selector.csv"), selector_rows,
              ["lambda", "selected_on_validation", "validation_cost_of_selection",
               "test_cost_of_selection", "selected_by_validation_macro_f1",
               "test_cost_of_macro_f1_selection", "test_optimal_diagnostic_only",
               "test_cost_of_diagnostic_optimum"])

    paired = [[algo, control, part, counts[(algo, part)]["missed_attack_cost"],
               counts[(control, part)]["missed_attack_cost"],
               counts[(algo, part)]["false_alarm_cost"],
               counts[(control, part)]["false_alarm_cost"]]
              for algo, control in controls.items() for part in PARTS]
    write_csv(os.path.join(args.outdir, "paired_no_host.csv"), paired,
              ["algorithm", "control", "part", "C_miss_with_host", "C_miss_no_host",
               "C_false_alarm_with_host", "C_false_alarm_no_host"])

    with open(os.path.join(args.outdir, "inputs_sha256.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "sha256"], *sorted(inputs.items())])

    figure(regions, val, test, config, os.path.join(args.outdir, "choice_regions.png"))

    # same metrics with the test matrices altered and validation untouched
    tampered = {v: {**m, "confusion_matrix": {"labels": m["confusion_matrix"]["labels"],
                                              "matrix": [[c * 3 + 5 for c in row]
                                                         for row in m["confusion_matrix"]["matrix"]]}}
                for v, m in metrics.items()}

    checks = {
        "recomputed costs equal the accepted metrics":
            all(counts[(v, "test")]["missed_attack_cost"] == m["cost"]["missed_attack_cost"]
                and counts[(v, "test")]["false_alarm_cost"] == m["cost"]["false_alarm_cost"]
                and counts[(v, "test")]["attack_confusions_unweighted"]
                == m["cost"]["mistriaged_attacks_not_weighted"] for v, m in metrics.items()),
        "missed, false alarms and attack confusions partition the errors":
            all(counts[(v, part)]["missed_attacks"] + counts[(v, part)]["false_alarms"]
                + counts[(v, part)]["attack_confusions_unweighted"]
                == sum(metrics[v][PARTS[part]]["matrix"][i][j]
                       for i in range(10) for j in range(10) if i != j)
                for v in VARIANTS for part in PARTS),
        "per-1000-row normalisation uses the rows of its own part":
            all(counts[(v, part)]["rows"] == metrics[v][f"{'validation' if part == 'validation' else 'test'}_rows"]
                for v in VARIANTS for part in PARTS),
        "changing only the test data leaves every choice on validation unchanged":
            {v: lines_of(tampered[v], "validation", costs) for v in pool} == val
            and {v: lines_of(tampered[v], "test", costs) for v in pool} != test
            and all(row[1] == select({v: lines_of(tampered[v], "validation", costs) for v in pool},
                                     pool, row[0]) for row in selector_rows),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'ERROR'} [risk02] {name}")
    print(f"     tables and figure written to {args.outdir}/")
    return int(not all(checks.values()))


def figure(regions, val, test, config, path):
    lam_max = 60.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (part, lines) in zip(axes, (("validation", val), ("test", test))):
        ax.set_ylim(0, 25000)
        ends = sorted(((a + b * lam_max, v) for v, (a, b) in lines.items()), reverse=True)
        shift = {v: 0 for _, v in ends}
        for (y0, v0), (y1, v1) in zip(ends, ends[1:]):
            if y0 - y1 < 900:
                shift[v1] = shift[v0] - 9
        for v, (a, b) in lines.items():
            ax.plot([0, lam_max], [a, a + b * lam_max], color=SERIES_COLOURS[v], linewidth=2,
                    label=v, zorder=3)
            ax.annotate(v, (lam_max, a + b * lam_max), xytext=(4, shift[v]), textcoords="offset points",
                        color=SERIES_COLOURS[v], fontsize=8, va="center")
        for lo, hi, v in regions[part]:
            if lo > 0:
                ax.axvline(lo, color="#9aa0a6", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                ax.annotate(f"λ={lo:.2f}", (lo, 24000), xytext=(3, 0),
                            textcoords="offset points", fontsize=8, color="#5f6368")
        ax.axvline(1, color="#3c4043", linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        ax.annotate("λ=1, configurazione accettata", (1, 0), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color="#3c4043")
        ax.set_title(f"C(λ) su {part} — preferito: "
                     + ", ".join(f"{v} per λ≥{lo:.2f}" for lo, _, v in regions[part]), fontsize=9)
        ax.set_xlabel("λ, peso del falso allarme rispetto all'attacco mancato")
        ax.set_xlim(0, lam_max * 1.12)
        ax.grid(True, color="#e8eaed", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("C(λ) in unità relative")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    sys.exit(main())
