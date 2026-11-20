"""The three figures of the common protocol v1, from the tables in reports_common/<dataset>/:
map of the choices, stability of the frontiers, validation-test errors by class. Main pool and
mean over seeds only; every dataset with tables is drawn.

    python scripts/cost_selection_figures.py
"""
from __future__ import annotations
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from xiiotid_cost_regions import SERIES_COLOURS

INK, MUTED, GRID = "#202124", "#5f6368", "#e8eaed"
ROOT = "reports_common"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tables():
    with open("configs/common_protocol_v1.json", encoding="utf-8") as f:
        inputs = json.load(f)["inputs"]
    with open(inputs, encoding="utf-8") as f:
        datasets = json.load(f)["datasets"]
    data = {}
    for key, ds in datasets.items():
        folder = os.path.join(ROOT, key)
        if os.path.exists(os.path.join(folder, "choices.csv")):
            data[ds["name"]] = {t: read(os.path.join(folder, f"{t}.csv"))
                                for t in ("choices", "frontiers", "radii", "normalized_counts")}
            data[ds["name"]]["pool"] = ds["pool"]
    return data


def it(x):
    return x.replace(".", ",")


def tidy(ax):
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def model_legend(fig, pool, extra=()):
    handles = [Patch(color=SERIES_COLOURS[m], label=m) for m in pool] + list(extra)
    ncol = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8)


def choice_map(data, path):
    panels = [("macro_f1", "0", "Macro-F1"),
              ("nominal", "0", "Costo nominale\n(= minimax a δ = 0)"),
              ("minimax_regret", "1/4", "Minimax regret\nδ = 0,25"),
              ("minimax_regret", "1/2", "Minimax regret\nδ = 0,5"),
              ("minimax_regret", "1", "Minimax regret\nδ = 1")]
    fig, axes = plt.subplots(len(data), len(panels), figsize=(12, 2.8 * len(data) + 0.6),
                             squeeze=False)
    tint = lambda c: tuple(0.62 + 0.38 * x for x in to_rgb(c))
    for row, (name, t) in zip(axes, data.items()):
        rows = [r for r in t["choices"] if r["pool"] == "main" and r["seed"] == "mean"]
        pis, lams = (sorted({r[k] for r in rows}, key=float) for k in ("pi", "lambda"))
        pick = {(r["policy"], r["delta"], r["pi"], r["lambda"]): r["model"] for r in rows}
        for ax, (policy, delta, title) in zip(row, panels):
            for (i, pi), (j, lam) in ((a, b) for a in enumerate(pis) for b in enumerate(lams)):
                m = pick[policy, delta, pi, lam]
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=tint(SERIES_COLOURS[m]),
                                       edgecolor="white", linewidth=2))
                ax.text(j + 0.5, i + 0.5, m, ha="center", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, len(lams))
            ax.set_ylim(0, len(pis))
            ax.set_xticks([j + 0.5 for j in range(len(lams))], lams, fontsize=8)
            ax.set_yticks([i + 0.5 for i in range(len(pis))], [it(p) for p in pis], fontsize=8)
            ax.tick_params(length=0)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("λ, costo FN / FP", fontsize=8, color=MUTED)
            for side in ax.spines.values():
                side.set_visible(False)
        row[0].set_ylabel(f"{name}\nπ, quota di attacchi", fontsize=9)
    model_legend(fig, next(iter(data.values()))["pool"])
    fig.suptitle("Modello scelto su Validation in ciascuno scenario", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stability(data, path):
    fig, axes = plt.subplots(1, len(data), figsize=(6.4 * len(data), 4.8), squeeze=False)
    for ax, (name, t) in zip(axes[0], data.items()):
        main = lambda rows: [r for r in rows if r["pool"] == "main"]
        scenarios = sorted({(r["pi"], r["lambda"]) for r in main(t["frontiers"])},
                           key=lambda s: (float(s[0]), float(s[1])))
        nominal = {(r["pi"], r["lambda"]): r["model"] for r in main(t["choices"])
                   if r["seed"] == "mean" and r["policy"] == "nominal" and r["delta"] == "0"}
        for idx, (pi, lam) in enumerate(scenarios):
            y = len(scenarios) - 1 - idx
            bars = (("validation", 0.16, 0.34, 1.0), ("test", -0.2, 0.22, 0.4))
            for part, dy, height, alpha in bars:
                for r in main(t["frontiers"]):
                    if (r["pi"], r["lambda"], r["part"]) != (pi, lam, part):
                        continue
                    lo, hi = float(r["delta_from_decimal"]), float(r["delta_to_decimal"])
                    ax.barh(y + dy, hi - lo, left=lo, height=height, alpha=alpha, linewidth=1,
                            color=SERIES_COLOURS[r["min_regret_model"]], edgecolor="white")
                    if part == "validation" and hi - lo >= 0.12:
                        ax.text((lo + hi) / 2, y + dy, r["min_regret_model"], fontsize=7,
                                ha="center", va="center", color=INK)
            radius = [float(r["radius_decimal"]) for r in main(t["radii"])
                      if (r["pi"], r["lambda"], r["part"]) == (pi, lam, "validation")
                      and r["model_m"] == nominal[pi, lam] and r["radius_decimal"]]
            if radius and min(radius) <= 1:
                ax.plot([min(radius)] * 2, [y - 0.02, y + 0.36], color=INK, linewidth=1.6)
        ax.set_yticks(range(len(scenarios)),
                      [f"π={it(pi)}  λ={lam}" for pi, lam in reversed(scenarios)], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("δ, variazione massima di ciascun peso", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="y", length=0)
        tidy(ax)
    extra = [Patch(color=MUTED, label="sopra: Validation (scelta)"),
             Patch(color=MUTED, alpha=0.4, label="sotto: Test (retrospettivo)"),
             Line2D([], [], color=INK, linewidth=1.6, label="raggio della scelta nominale")]
    model_legend(fig, next(iter(data.values()))["pool"], extra)
    fig.suptitle("Modello a minimax regret al variare di δ", fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def errors_by_class(data, path):
    n = sum(len({r["class"] for r in t["normalized_counts"]}) for t in data.values())
    fig, axes = plt.subplots(len(data), 2, figsize=(11, 0.42 * n + 1.6 * len(data)), squeeze=False)
    for row, (name, t) in zip(axes, data.items()):
        pool = t["pool"]
        classes = list(dict.fromkeys(r["class"] for r in t["normalized_counts"]))
        values = {}
        for r in t["normalized_counts"]:
            for kind, v in (("missed", float(r["q"]) * float(r["missed_rate"])),
                            ("alarm", float(r["false_alarm_rate"]))):
                values.setdefault((r["model"], r["part"], r["class"], kind), []).append(v)
        mean = lambda *key: sum(values[key]) / len(values[key])
        for ax, kind, title, unit in (
                (row[0], "missed", "Attacchi mancati: q_i × tasso attacco→benigno",
                 "per unità di λ·π"),
                (row[1], "alarm", "Falsi allarmi: tasso benigno→classe i", "per unità di (1 − π)")):
            for k, m in enumerate(pool):
                offset = (k - (len(pool) - 1) / 2) * 0.24
                for i, c in enumerate(classes):
                    y, colour = len(classes) - 1 - i + offset, SERIES_COLOURS[m]
                    v, te = mean(m, "validation", c, kind), mean(m, "test", c, kind)
                    ax.plot([v, te], [y, y], color=colour, linewidth=1.2)
                    ax.plot(v, y, "o", color=colour, markersize=5)
                    ax.plot(te, y, "o", markerfacecolor="white", markeredgecolor=colour,
                            markersize=5)
            ax.set_yticks(range(len(classes)), list(reversed(classes)), fontsize=8)
            ax.set_xlim(left=0)
            ax.set_title(f"{name} — {title}", fontsize=9)
            ax.set_xlabel(f"contributo al costo atteso {unit}", fontsize=8)
            ax.tick_params(axis="y", length=0)
            tidy(ax)
    extra = [Line2D([], [], marker="o", linestyle="", color=MUTED, label="Validation"),
             Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                    markeredgecolor=MUTED, label="Test")]
    model_legend(fig, next(iter(data.values()))["pool"], extra)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    data = load_tables()
    if not data:
        print(f"ERROR no tables in {ROOT}/: run scripts/cost_selection.py --dataset first")
        return 1
    out = os.path.join(ROOT, "figures")
    os.makedirs(out, exist_ok=True)
    choice_map(data, os.path.join(out, "choice_map.png"))
    stability(data, os.path.join(out, "stability.png"))
    errors_by_class(data, os.path.join(out, "errors_by_class.png"))
    print(f"figures for {', '.join(data)} written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
