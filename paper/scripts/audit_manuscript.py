"""Check frozen manuscript evidence without fitting models or redrawing figures.

Usage: python paper/scripts/audit_manuscript.py [--local-archive PATH]
The optional archive checks historical source identities not shipped in this repo.
Git comparisons run only when the named release commits are available locally.
DOI metadata is checked against its dated cache, not refreshed from the network.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import re
import shutil
import subprocess

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
BASELINE_MANUSCRIPT = "1b6ac02dd85f43cea02c893c6f193b7b313a0594"  # manuscript-v4
NUMERIC_KEYS = ("spectral_leakage_fraction", "unmatched_mass_fraction",
                "test_clean_rmse_sigma", "w1_logD")
METHODS = ("RAI", "RAI_S", "DOME", "DOME_S")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def close(actual, expected, context):
    require(bool(np.allclose(actual, expected, rtol=1e-11, atol=1e-13)), context)


def aggregate(rows):
    return dict(n=len(rows),
                correct=sum(row["correct_rank"] == "True" for row in rows),
                recovered=sum(row["correct_rank"] == "True" and
                              int(row["matched_components"]) == int(row["r_true"])
                              for row in rows),
                **{key: mean(rows, key) for key in NUMERIC_KEYS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-archive", type=Path)
    args = parser.parse_args()
    with (ROOT / "paper/evidence/metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cached = json.loads((ROOT / "paper/evidence/manuscript_aggregation.json").read_text())
    positive = {method: sorted((row for row in rows if row["partition"] == "test"
                               and row["scenario"] not in ("null", "broad")
                               and row["method"] == method), key=lambda row: row["id"])
                for method in METHODS}
    stats = {method: aggregate(selected) for method, selected in positive.items()}
    table = (ROOT / "paper/sections/table_positive.tex").read_text()
    for method, stat in stats.items():
        require(stat["n"] == 36, f"Unexpected primary population: {method}")
        for key, value in stat.items():
            close(value, cached["positive_test"][method][key], f"Primary mean/count: {method}/{key}")
        table_row = (f"{method.replace('_', '-')} & {stat['correct']}/36 & {stat['recovered']}/36"
                     f" & {100*stat['spectral_leakage_fraction']:.3f}"
                     f" & {100*stat['unmatched_mass_fraction']:.3f}"
                     f" & {stat['test_clean_rmse_sigma']:.4f} & {stat['w1_logD']:.4f}")
        require(table_row in table, f"Table transcription: {method}")
    paired = {}
    for method in ("RAI", "DOME"):
        old, new = positive[method], positive[method + "_S"]
        require([row["id"] for row in old] == [row["id"] for row in new], "Unpaired cases")
        paired[method] = {}
        for key in ("spectral_leakage_fraction", "test_clean_rmse_sigma", "w1_logD"):
            delta = np.array([float(b[key]) - float(a[key]) for a, b in zip(old, new)])
            rng = np.random.default_rng(9925)
            interval = np.quantile(delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1), [.025, .975])
            expected = cached["paired_changes_positive"][method][key]
            close(delta.mean(), expected["delta"], f"Paired change: {method}/{key}")
            close(interval, expected["bootstrap95"], f"Bootstrap: {method}/{key}")
            paired[method][key] = dict(delta=float(delta.mean()), bootstrap95=interval.tolist())

    numerical_success = {}
    for method in METHODS:
        for panel, count in (("test", 48), ("archive", 24)):
            selected = [row for row in rows if row["method"] == method and row["partition"] == panel]
            require(len(selected) == count and all(row["success"] == "True" for row in selected),
                    f"Atomic success flags: {method}/{panel}")
            numerical_success[f"{method}/{panel}"] = count
    neural = {}
    results_tex = (ROOT / "paper/sections/results.tex").read_text()
    for method in ("RAI_Net_original", "RAI_Net_v2"):
        neural[method] = {}
        for panel in ("test", "archive"):
            selected = [row for row in rows if row["method"] == method and row["partition"] == panel]
            require(len(selected) == 24, f"Neural population: {method}/{panel}")
            if panel == "test":
                require(set(collections.Counter(row["scenario"] for row in selected).values()) == {3},
                        "Neural scenario balance")
            failed = [row["id"] for row in selected if row["success"] != "True"]
            require(failed == ([] if panel == "test" else ["r1_snr50_rep2"]), "Neural failures changed")
            stat = dict(n=len(selected), w1_logD=mean(selected, "w1_logD"),
                        test_clean_rmse_sigma=mean(selected, "test_clean_rmse_sigma"),
                        failed_cases=failed)
            for key in ("w1_logD", "test_clean_rmse_sigma"):
                require(f"{stat[key]:.6f}" in results_tex, f"Neural table value: {method}/{panel}/{key}")
            neural[method][panel] = stat

    immutable = json.loads((ROOT / "provenance/source_hashes.json").read_text())
    for entry in immutable:
        require(sha(ROOT / entry["path"]) == entry["sha256"], f"Immutable source: {entry['path']}")
    manifest = json.loads((ROOT / "paper/evidence/method_formulations.json").read_text())
    main_tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    sections = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "paper/sections").glob("*.tex"))
    labels = re.findall(r"\\label\{([^}]+)\}", sections)
    require(len(set(labels)) == len(labels), "Duplicate equation/theorem labels")
    require(len(re.findall(r"\\begin\{algorithm\}", sections)) == 1, "Algorithm 1 environment")
    checked_sources, skipped_sources = [], []
    for entry in manifest["sources"]:
        require(all(label in labels for label in entry["latex_labels"]), f"Source/equation map: {entry['method']}")
        base = ROOT if entry["source_scope"] == "public" else args.local_archive
        if base is None:
            skipped_sources.append(entry["path"])
            continue
        require(sha(base / entry["path"]) == entry["sha256"], f"Formulation source: {entry['path']}")
        checked_sources.append(entry["path"])

    refs_tex = (ROOT / "paper/references.tex").read_text(encoding="utf-8")
    references = set(re.findall(r"\\bibitem\{([^}]+)\}", refs_tex))
    citations = set(key for group in re.findall(r"\\cite\{([^}]+)\}", main_tex + sections) for key in group.split(","))
    require(references == citations and len(references) == 48, "Reference/citation coverage")
    doi_cache = json.loads((ROOT / "paper/evidence/reference_metadata.json").read_text(encoding="utf-8"))
    dois = set(re.findall(r"https://doi.org/([^}]+)", refs_tex))
    cached_dois = {entry["doi"] for entry in doi_cache["references"] if entry["status"] == "verified_registered_metadata"}
    require(dois == cached_dois and len(dois) == 46, "Dated DOI-cache coverage")
    backmatter = (ROOT / "paper/sections/backmatter.tex").read_text(encoding="utf-8")
    require("Acknowledgment" not in backmatter, "Acknowledgments must be absent")
    require("proposal based on author order" in backmatter and "must be confirmed by all three authors" in backmatter,
            "Provisional CRediT qualification")
    require("cannot be inferred" in backmatter and "no conflict" not in backmatter.lower(), "Unconfirmed conflicts")
    require("OpenAI Codex (GPT-6)" in sections, "AI-use disclosure")
    author_metadata = json.loads((ROOT / "paper/evidence/author_metadata.json").read_text(encoding="utf-8"))
    author_names = [entry["name"] for entry in author_metadata["authors"]]
    require(author_names == ["Victor Valdivieso", "Ignacio Fernández", "Francisco Manuel Arrabal-Campos"], "Author order")
    for grant in author_metadata["funding_identifiers"]:
        require(grant in backmatter, f"Funding identifier: {grant}")
    for entry in author_metadata["authors"]:
        require(entry["email"] in main_tex, f"Author email: {entry['name']}")
        if entry["orcid"]:
            total = 0
            compact = entry["orcid"].replace("-", "")
            for char in compact[:-1]:
                total = (total + int(char)) * 2
            checksum = (12 - total % 11) % 11
            require(compact[-1] == ("X" if checksum == 10 else str(checksum)), "ORCID checksum")

    pdf = ROOT / "paper/output/pdf/Positive_Joint_Laplace_Inversion_Mathematics.pdf"
    reader = PdfReader(pdf)
    texts = [page.extract_text() for page in reader.pages]
    require(len(texts) == 38, "Expected 38-page v5 PDF")
    require(all(name in reader.metadata.author for name in author_names), "PDF author metadata")
    pdf_links = []
    for page in reader.pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if action.get("/URI"):
                pdf_links.append(action["/URI"])
    for entry in author_metadata["authors"]:
        if entry["orcid"]:
            require(any(entry["orcid"] in link for link in pdf_links), "PDF ORCID hyperlink")
    require("Acknowledgments" not in "\n".join(texts), "PDF acknowledgments removal")
    require("manuscript-v5" in "\n".join(texts), "PDF release citation")
    log = ROOT / "paper/output/pdf/main.log"
    log_checked = log.exists()
    if log_checked:
        log_text = log.read_text(errors="replace")
        require(not any(token in log_text for token in ("Overfull", "undefined", "duplicate ignored", "! LaTeX Error")),
                "LaTeX warnings/errors")

    git_checks = {"status": "skipped: baseline Git objects unavailable"}
    have_baselines = bool(shutil.which("git")) and all(
        subprocess.run(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=ROOT, capture_output=True).returncode == 0
        for ref in (BASELINE_MANUSCRIPT, "v0.1.0"))
    if have_baselines:
        old_audit = json.loads(subprocess.check_output(["git", "show", f"{BASELINE_MANUSCRIPT}:verification/paper_audit.json"], cwd=ROOT))
        numerical = old_audit["numerical_files_unchanged_from_v0_1_0"]
        for relative in numerical:
            expected = subprocess.check_output(["git", "show", f"v0.1.0:{relative}"], cwd=ROOT)
            require((ROOT / relative).read_bytes() == expected, f"Changed numerical file: {relative}")
        figures = list((ROOT / "paper/figures").glob("*"))
        for path in figures + [ROOT / "paper/references.tex", ROOT / "paper/evidence/metrics.csv"]:
            relative = path.relative_to(ROOT).as_posix()
            expected = subprocess.check_output(["git", "show", f"{BASELINE_MANUSCRIPT}:{relative}"], cwd=ROOT)
            require(path.read_bytes() == expected, f"Changed frozen figure/reference/metric: {relative}")
        git_checks = dict(status="passed", numerical_files_unchanged_from_v0_1_0=len(numerical),
                          figure_files_unchanged_from_manuscript_v4=len(figures),
                          bibliography_and_metrics_unchanged_from_manuscript_v4=True)

    algebra = json.loads((ROOT / "verification/formulation_checks.json").read_text())
    require(algebra["passed"] == 27, "Algebra check count")
    report = dict(revision="manuscript-v5", scope="Saved-data reaggregation, source identities, manuscript structure; no solver fits, no new statistical validation.",
                  primary=stats, paired_changes=paired, atomic_success_flags=numerical_success, neural=neural,
                  immutable_source_files=len(immutable), formulation_sources_checked=checked_sources,
                  formulation_sources_not_available=skipped_sources, references=len(references),
                  doi_cache_entries=len(dois), doi_cache_checked_utc=doi_cache["checked_utc"],
                  doi_network_refresh_performed=False, algebra_checks=algebra["passed"],
                  author_metadata_checks="passed; authorship roles and conflicts remain unconfirmed",
                  pages=len(texts), pdf_sha256=sha(pdf), latex_log_checked=log_checked,
                  visual_review="Separate human/model inspection; not inferred from these automated checks.",
                  git_comparison=git_checks)
    target = ROOT / "verification/manuscript_audit_v5.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(target), "pages": len(texts),
                      "primary_cases": 36, "formulation_sources_checked": len(checked_sources),
                      "algebra_checks": algebra["passed"], "git": git_checks}))


if __name__ == "__main__":
    main()
