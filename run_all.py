"""
run_all.py — Batch evaluator for all 30 chart folders.

Usage:
    python run_all.py
    python run_all.py --input_dir test_data --output_dir output

Directory structure expected:
    test_data/
        chart_01/
            chart_01.json
            chart_01.md        ← ground truth (auto-discovered by test.py logic)
        chart_02/
            chart_02.json
            chart_02.md
        ...

Output:
    output/
        chart_01.json          ← one evaluation report per input file
        chart_02.json
        ...
    output/summary.json        ← aggregate across all 30 files
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from collections import defaultdict

# Import directly — no subprocess overhead
from evaluator import ClinicalEntityEvaluator


# ── MD discovery (mirrors logic in test.py) ───────────────────────────────────

def find_md(json_path: Path) -> str:
    """Return content of the .md ground-truth file next to the JSON, or ''."""
    candidates = [
        json_path.with_suffix(".md"),
        *json_path.parent.glob("*.md"),
        json_path.parent.parent / (json_path.parent.name + ".md"),
    ]
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8", errors="replace")
    return ""


# ── Safe average that ignores None values ─────────────────────────────────────

def _safe_avg(values: list) -> float | None:
    """Average a list that may contain None. Returns None if all values are None."""
    real = [v for v in values if v is not None]
    return round(sum(real) / len(real), 4) if real else None


# ── Aggregate summary across all output files ─────────────────────────────────

def build_summary(results: list[dict]) -> dict:
    """
    For each metric key, average across files — skipping None (= not present).
    Produces one summary dict in the same schema as individual output files.
    """
    # Collect all values per key
    entity_type_vals  = defaultdict(list)
    assertion_vals    = defaultdict(list)
    temporality_vals  = defaultdict(list)
    subject_vals      = defaultdict(list)
    date_acc_vals     = []
    completeness_vals = []

    for r in results:
        for k, v in r.get("entity_type_error_rate", {}).items():
            entity_type_vals[k].append(v)
        for k, v in r.get("assertion_error_rate", {}).items():
            assertion_vals[k].append(v)
        for k, v in r.get("temporality_error_rate", {}).items():
            temporality_vals[k].append(v)
        for k, v in r.get("subject_error_rate", {}).items():
            subject_vals[k].append(v)
        if r.get("event_date_accuracy") is not None:
            date_acc_vals.append(r["event_date_accuracy"])
        if r.get("attribute_completeness") is not None:
            completeness_vals.append(r["attribute_completeness"])

    return {
        "files_evaluated": len(results),
        "entity_type_error_rate":  {k: _safe_avg(v) for k, v in entity_type_vals.items()},
        "assertion_error_rate":    {k: _safe_avg(v) for k, v in assertion_vals.items()},
        "temporality_error_rate":  {k: _safe_avg(v) for k, v in temporality_vals.items()},
        "subject_error_rate":      {k: _safe_avg(v) for k, v in subject_vals.items()},
        "event_date_accuracy":     _safe_avg(date_acc_vals),
        "attribute_completeness":  _safe_avg(completeness_vals),
    }


# ── Pretty progress printer ───────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch clinical entity evaluator")
    parser.add_argument("--input_dir",  default="test_data", help="Root folder with chart subdirectories")
    parser.add_argument("--output_dir", default="output",    help="Folder to write per-file JSON reports")
    args = parser.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"[!] Input directory not found: {input_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # ── Discover all JSON files ───────────────────────────────────────────────
    # Each subfolder has exactly one JSON matching the folder name, e.g.:
    #   test_data/chart_01/chart_01.json
    # We also handle flat layouts where JSONs sit directly in input_root.
    json_files = sorted(input_root.rglob("*.json"))

    # Exclude any output/ files accidentally inside the tree
    json_files = [f for f in json_files if output_root not in f.parents]

    if not json_files:
        print(f"[!] No JSON files found under {input_root}")
        sys.exit(1)

    total = len(json_files)
    print(f"\n{'='*60}")
    print(f"  Clinical Entity Batch Evaluator")
    print(f"  Input  : {input_root.resolve()}")
    print(f"  Output : {output_root.resolve()}")
    print(f"  Files  : {total}")
    print(f"{'='*60}\n")

    evaluator = ClinicalEntityEvaluator()
    results   = []
    failed    = []

    for idx, jf in enumerate(json_files, start=1):
        out_file = output_root / jf.name
        print(f"{_bar(idx-1, total)}  {jf.name}", end="\r")

        try:
            # ── Load JSON ────────────────────────────────────────────────────
            raw = json.loads(jf.read_text(encoding="utf-8"))
            entities = raw if isinstance(raw, list) else raw.get("entities", [])

            # ── Load MD ground truth ─────────────────────────────────────────
            chart_md = find_md(jf)
            md_status = f"MD:{len(chart_md):,}chars" if chart_md else "MD:missing"

            # ── Evaluate ─────────────────────────────────────────────────────
            result = evaluator.evaluate(jf.stem, entities, chart_md=chart_md)

            # ── Write individual output ──────────────────────────────────────
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

            overall_err = result["entity_type_error_rate"].get("_overall", "?")
            print(f"{_bar(idx, total)}  {jf.name}  [{md_status}  entity_err={overall_err}]")

            results.append(result)

        except Exception as e:
            print(f"\n[!] FAILED: {jf.name} — {e}")
            traceback.print_exc()
            failed.append({"file": str(jf), "error": str(e)})

    # ── Write summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Completed: {len(results)}/{total}  |  Failed: {len(failed)}")
    print(f"{'='*60}")

    if results:
        summary = build_summary(results)
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[+] Summary written → {summary_path}")

        # Print key aggregate metrics to console
        print("\n── Aggregate error rates (avg across all files) ──────────")
        print(f"  Overall entity type error : {summary['entity_type_error_rate'].get('_overall')}")
        print(f"  Overall assertion error   : {summary['assertion_error_rate'].get('_overall')}")
        print(f"  Overall temporality error : {summary['temporality_error_rate'].get('_overall')}")
        print(f"  Overall subject error     : {summary['subject_error_rate'].get('_overall')}")
        print(f"  Event date accuracy       : {summary['event_date_accuracy']}")
        print(f"  Attribute completeness    : {summary['attribute_completeness']}")

        print("\n── Entity type breakdown ──────────────────────────────────")
        for k, v in summary["entity_type_error_rate"].items():
            if k != "_overall":
                bar = ("█" * int((v or 0) * 20)).ljust(20)
                print(f"  {k:20s} {bar} {v}")

    if failed:
        fail_path = output_root / "failed.json"
        fail_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"\n[!] {len(failed)} failed files logged → {fail_path}")


if __name__ == "__main__":
    main()