"""Stream JSONL cleaning with disk-backed exact normalized-document dedup."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unicodedata


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl_texts(path, text_field="text"):
    """Yield text while reporting malformed input with file and line context."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: line {line_number}: invalid JSON") from error
            if not isinstance(record, dict) or not isinstance(record.get(text_field), str):
                raise ValueError(f"{path}: line {line_number}: field {text_field!r} must be a string")
            yield record[text_field]


def prepare_jsonl(input_path, output_dir, min_chars=80, val_fraction=0.1, text_field="text", validation_input=None):
    """Normalize, deduplicate, and split real documents; return a data report.

    With ``validation_input`` the original train/validation membership is kept,
    and validation duplicates overlapping train are removed. Otherwise SHA-256
    chooses membership after exact global deduplication. Empty partitions raise
    an error: no document is moved to silently repair a tiny split.
    """
    if not isinstance(min_chars, int) or isinstance(min_chars, bool) or min_chars < 1:
        raise ValueError("min_chars must be a positive integer")
    if not isinstance(val_fraction, (int, float)) or isinstance(val_fraction, bool) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a nonempty string")
    input_path, output_dir = Path(input_path), Path(output_dir)
    sources = [(input_path, "train")]
    if validation_input is not None:
        sources.append((Path(validation_input), "val"))
    for path, _ in sources:
        if path.resolve() in {(output_dir / name).resolve() for name in ("train.jsonl", "val.jsonl", "data_report.json")}:
            raise ValueError("Input cannot also be an output file; choose a separate output directory")
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"format_version": 1, "normalization": "NFC; CRLF/CR to LF; outer whitespace stripped",
              "deduplication": "exact normalized text before splitting (SQLite UNIQUE)",
              "split_method": "original partitions" if validation_input is not None else "SHA-256 threshold",
              "min_chars": min_chars, "val_fraction": None if validation_input is not None else val_fraction,
              "text_field": text_field, "input_documents": 0, "rejected_short": 0, "duplicates": 0,
              "validation_overlap_removed": 0, "train_documents": 0, "val_documents": 0,
              "sources": [{"path": str(path), "sha256": file_sha256(path), "partition": partition} for path, partition in sources]}
    with tempfile.TemporaryDirectory(prefix="fm-clean-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        connection = sqlite3.connect(temporary / "seen.sqlite")
        connection.execute("CREATE TABLE seen (text TEXT PRIMARY KEY, partition TEXT NOT NULL)")
        try:
            with (temporary / "train.jsonl").open("w", encoding="utf-8") as train, (temporary / "val.jsonl").open("w", encoding="utf-8") as validation:
                outputs = {"train": train, "val": validation}
                for path, original_partition in sources:
                    for text in iter_jsonl_texts(path, text_field):
                        report["input_documents"] += 1
                        text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
                        if len(text) < min_chars:
                            report["rejected_short"] += 1
                            continue
                        document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        partition = original_partition if validation_input is not None else ("val" if int(document_id[:16], 16) / 2**64 < val_fraction else "train")
                        try:
                            connection.execute("INSERT INTO seen VALUES (?, ?)", (text, partition))
                        except sqlite3.IntegrityError:
                            report["duplicates"] += 1
                            previous = connection.execute("SELECT partition FROM seen WHERE text = ?", (text,)).fetchone()[0]
                            if original_partition == "val" and previous == "train":
                                report["validation_overlap_removed"] += 1
                            continue
                        outputs[partition].write(json.dumps({"text": text, "document_id": document_id}, ensure_ascii=False) + "\n")
                        report[f"{partition}_documents"] += 1
            if not report["train_documents"] or not report["val_documents"]:
                raise ValueError("Both train and validation must be nonempty after cleaning. Supply more distinct documents, lower min_chars, change val_fraction, or provide --validation-input with an original validation partition.")
            for name in ("train.jsonl", "val.jsonl"):
                (temporary / name).replace(output_dir / name)
            report["outputs"] = {name: {"sha256": file_sha256(output_dir / name)} for name in ("train.jsonl", "val.jsonl")}
            (output_dir / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            connection.close()
    return report
