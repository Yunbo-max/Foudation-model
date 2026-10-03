"""Fit user-provided measurements; never ships or creates pretend results."""
import argparse
import json
from pathlib import Path

from fm_tutorial.data import file_sha256
from fm_tutorial.scaling import fit_power_law, read_measurements


def main():
    parser = argparse.ArgumentParser(description="Fit loss ~ axis**exponent on real measured CSV, holding the other axis fixed. Required columns: params,tokens,loss; optional split=train|holdout.")
    parser.add_argument("--csv", required=True, help="CSV of actual measured runs using consistent loss units")
    parser.add_argument("--axis", choices=("tokens", "params"), default="tokens")
    parser.add_argument("--output", help="Optional output JSON path (default: print only)")
    args = parser.parse_args()
    try:
        result = fit_power_law(read_measurements(args.csv), args.axis)
        result["source"] = {"path": str(args.csv), "sha256": file_sha256(args.csv)}
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(rendered, end="")


if __name__ == "__main__":
    main()
