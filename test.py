"""
test.py — Clinical AI Pipeline Evaluator
HiLabs Workshop

Usage:
    python test.py input.json output.json

The script auto-discovers the .md ground-truth file by looking in the
same directory as the JSON (matching the dataset folder structure):

    test_data/
      chart_01/
        chart_01.json   ← input
        chart_01.md     ← OCR ground truth (auto-discovered)

    output/
        chart_01.json   ← output written here
"""

import json
import sys
import os
from pathlib import Path
from evaluator import ClinicalEntityEvaluator


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_md(json_path: str) -> str:
    """
    Find the .md file that accompanies the JSON.
    Tries (in order):
      1. Same directory, same stem:  chart_01/chart_01.md
      2. Same directory, any .md:    chart_01/*.md
      3. Parent directory, same stem: chart_01.md next to the folder
    Returns the file content, or empty string if not found.
    """
    p = Path(json_path)
    candidates = [
        p.with_suffix(".md"),                          # same dir, same name
        *p.parent.glob("*.md"),                        # same dir, any .md
        p.parent.parent / (p.parent.name + ".md"),    # one level up
    ]
    for c in candidates:
        if c.exists():
            print(f"[+] Ground truth MD: {c}")
            return c.read_text(encoding="utf-8", errors="replace")

    print("[!] No .md ground truth found — LLM will evaluate without chart context")
    return ""


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    if len(sys.argv) < 3:
        print("Usage: python test.py input.json output.json")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    print(f"[+] Loading JSON : {input_path}")
    data = load_json(input_path)

    # Load the accompanying .md as ground truth
    chart_md = load_md(input_path)

    file_name = Path(input_path).stem
    evaluator = ClinicalEntityEvaluator()

    entities = data if isinstance(data, list) else data.get("entities", [])
    print(f"[+] Evaluating {len(entities)} entities...")
    result = evaluator.evaluate(file_name, entities, chart_md=chart_md)

    save_json(result, output_path)
    print(f"[+] Output written: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()