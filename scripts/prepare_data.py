"""Prepare actual local JSONL text; no built-in dataset is generated."""
import argparse
import json

from fm_tutorial.data import prepare_jsonl


def main():
    parser = argparse.ArgumentParser(description="Normalize and deduplicate real JSONL documents before a deterministic split.")
    parser.add_argument("--input", required=True, help="Training source JSONL with a text field")
    parser.add_argument("--out-dir", default="data/clean")
    parser.add_argument("--validation-input", help="Original validation JSONL; preserves original partitions and removes validation/train duplicates")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Hash split fraction, used only without --validation-input")
    args = parser.parse_args()
    try:
        report = prepare_jsonl(args.input, args.out_dir, args.min_chars, args.val_fraction, args.text_field, validation_input=args.validation_input)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
