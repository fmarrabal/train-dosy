"""Regenerate Figure 2 typography from frozen metrics; never execute a solver.

Run with the repository Python environment. Optional --metrics and
--original-generator paths allow reuse after moving this standalone script.
The 24 plotted counts are checked against the original aggregate function,
extracted with ast rather than running the original figure-preparation script.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["RAI", "RAI_S", "DOME", "DOME_S"]
SCENARIOS = ["single", "double", "triple", "close", "weak", "proportional"]
COLORS = ["#96afca", "#176b96", "#d9ae8e", "#ad5a20"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    script_directory = Path(__file__).resolve().parent
    root = script_directory.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=root / "validation/primary/atomic/metrics.csv")
    parser.add_argument("--original-generator", type=Path,
                        default=root / "validation/primary/replay_figures.py")
    parser.add_argument("--output", type=Path, default=script_directory / "generated/recovery")
    args = parser.parse_args()
    here = args.output.resolve()
    here.mkdir(parents=True, exist_ok=True)
    original_hash = sha(args.original_generator)
    metrics_hash = sha(args.metrics)
    with args.metrics.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["partition"] == "test"
            and row["scenario"] not in ("null", "broad")
            and row["method"] in METHODS]

    # Compile only the original pure aggregation function. No plotting entry
    # point, reconstruction, or benchmark from that source is executed.
    original_tree = ast.parse(args.original_generator.read_text(encoding="utf-8"))
    aggregate_nodes = [node for node in ast.walk(original_tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "aggregate"]
    assert len(aggregate_nodes) == 1
    namespace = {"np": np}
    module = ast.Module(body=[aggregate_nodes[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(args.original_generator), "exec"), namespace)
    original_aggregate = namespace["aggregate"]

    counts = {}
    provenance_rows = []
    for method in METHODS:
        counts[method] = []
        for scenario in SCENARIOS:
            subset = [row for row in rows
                      if row["method"] == method and row["scenario"] == scenario]
            assert len(subset) == 6
            recovered_ids = [row["id"] for row in subset
                             if row["correct_rank"] == "True"
                             and int(row["matched_components"]) == int(row["r_true"])]
            count = len(recovered_ids)
            original_count = original_aggregate(subset)["recovered"]
            assert count == original_count
            counts[method].append(count)
            provenance_rows.append(dict(
                method=method, scenario=scenario, recovered=count,
                denominator=len(subset), original_recovered=original_count,
                agrees_with_original=True,
                recovered_problem_ids=";".join(recovered_ids),
                metrics_sha256=metrics_hash,
                original_generator_sha256=original_hash))
    assert len(provenance_rows) == 24

    # Same canvas, bar geometry, colors, ordering, limits, and mathematical
    # endpoint as the original. Only text sizes and ylabel line breaking change.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    fig, ax = plt.subplots(figsize=(7.2, 2.65), layout="constrained")
    for index, method in enumerate(METHODS):
        ax.bar(np.arange(6) + (index - 1.5) * .18, counts[method], width=.17,
               label=method.replace("_", "-"), color=COLORS[index])
    ax.set(xticks=np.arange(6), xticklabels=[value.title() for value in SCENARIOS],
           ylim=(0, 7.6), yticks=[0, 2, 4, 6],
           ylabel="Correct order and all rates\n(out of 6)")
    ax.legend(ncol=4, loc="upper center", frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("pdf", "png"):
        fig.savefig(here / f"recovery.{extension}")
    plt.close(fig)

    with (here / "recovery_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance_rows[0]))
        writer.writeheader()
        writer.writerows(provenance_rows)
    assert sha(args.metrics) == metrics_hash
    assert sha(args.original_generator) == original_hash
    provenance = dict(
        change="Recovery counts and figure replayed from saved numerical outcomes.",
        original_generator=args.original_generator.resolve().relative_to(root).as_posix() if args.original_generator.resolve().is_relative_to(root) else args.original_generator.name,
        original_generator_sha256=original_hash,
        metrics_source=args.metrics.resolve().relative_to(root).as_posix() if args.metrics.resolve().is_relative_to(root) else args.metrics.name, metrics_sha256=metrics_hash,
        counts_checked=24, all_counts_equal_original=True,
        canvas_inches=[7.2, 2.65], bar_width=.17, bar_spacing=.18,
        methods=METHODS, scenarios=SCENARIOS, colors=COLORS,
        numerical_refits=0,
        outputs={name: sha(Path(__file__)) if name == "plot_recovery.py" else sha(here / name) for name in
                 ("recovery.pdf", "recovery.png", "recovery_counts.csv",
                  "plot_recovery.py")})
    (here / "recovery_replay.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(dict(verified_counts=24, counts=counts, numerical_refits=0), indent=2))


if __name__ == "__main__":
    main()
