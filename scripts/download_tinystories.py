"""Stream the original published TinyStories text files with explicit closure.

The dataset README designates TinyStories-train.txt and TinyStories-valid.txt
as the original paper's data. These are distinct from TinyStoriesV2-GPT4 and
from the Hub's Parquet representation, which changes paragraph spacing. Direct
HTTP streaming avoids native Parquet scan threads during partial-read shutdown;
only the prefix needed for the requested documents is consumed.
"""
import argparse
import codecs
import json
from pathlib import Path
import tempfile

from fm_tutorial.data import file_sha256


DATASET = "roneneldan/TinyStories"
SOURCE_FILES = {"train": "TinyStories-train.txt", "validation": "TinyStories-valid.txt"}
MARKER = "<|endoftext|>"


def iter_original_documents(byte_chunks):
    """Parse complete marker-delimited UTF-8 stories across arbitrary chunks.

    Internal line breaks are retained. Outer whitespace and separator lines are
    removed. Reject malformed UTF-8 and an unterminated tail rather than saving
    a silently truncated story. Closing the generator ends a partial read.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    pending = ""
    lines = []
    for chunk in byte_chunks:
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if line.rstrip("\r") == MARKER:
                text = "".join(lines).strip()
                lines = []
                if text:
                    yield text
            else:
                lines.append(line + "\n")
    pending += decoder.decode(b"", final=True)
    if pending.rstrip("\r") == MARKER:
        text = "".join(lines).strip()
        lines = []
        pending = ""
        if text:
            yield text
    if ("".join(lines) + pending).strip():
        raise ValueError("Original TinyStories source has an unterminated/truncated final story")


def write_partition(documents, path, limit, split):
    """Write at most ``limit`` complete documents and close an early iterator."""
    iterator = iter(documents)
    count = 0
    try:
        with Path(path).open("w", encoding="utf-8") as output:
            while count < limit:
                try:
                    text = next(iterator)
                except StopIteration:
                    break
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Upstream {split} document {count + 1} is not a nonempty text string")
                output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if not count:
        raise ValueError(f"Upstream {split} partition is empty")
    return {"documents": count, "requested_limit": limit, "sha256": file_sha256(path)}


def download_partition(client, url, path, limit, split):
    bytes_read = 0
    with client.stream("GET", url, headers={"Range": "bytes=0-", "Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        # A server may ignore Range and return 200; streamed consumption is
        # still bounded by the document limit and the response closes below.
        if response.status_code == 206 and not response.headers.get("content-range", "").startswith("bytes 0-"):
            raise ValueError("Source server returned a range that does not start at byte zero")

        def chunks():
            nonlocal bytes_read
            for chunk in response.iter_bytes(chunk_size=65536):
                bytes_read += len(chunk)
                yield chunk

        report = write_partition(iter_original_documents(chunks()), path, limit, split)
        report["bytes_read"] = bytes_read
        report["http_status"] = response.status_code
        report["content_range"] = response.headers.get("content-range")
    return report


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("document limit must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description="Stream real stories from the original published TinyStories train/validation text files, preserving both partitions and closing partial HTTP reads.")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--max-train", type=positive_integer, default=20000)
    parser.add_argument("--max-validation", type=positive_integer, default=2000)
    parser.add_argument("--revision", default="main", help="Hub revision; resolved commit SHA is recorded and used for loading")
    args = parser.parse_args()
    try:
        import httpx
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError:
        parser.error('Install optional dependencies with: pip install -e ".[data]"')
    out_dir = Path(args.out_dir)
    # Resolve and freeze the upstream revision before opening either partition.
    info = HfApi().dataset_info(DATASET, revision=args.revision, files_metadata=True)
    revision = info.sha
    if not revision:
        parser.error("Hub did not return a dataset commit SHA; cannot record reproducible provenance")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": DATASET, "source": f"https://huggingface.co/datasets/{DATASET}",
                "requested_revision": args.revision, "resolved_revision": revision,
                "dataset_version": "original TinyStories (GPT-3.5 and GPT-4), as designated by the dataset README",
                "representation": "published UTF-8 text, split on standalone <|endoftext|> lines; outer whitespace stripped, internal paragraphs retained",
                "transport": "bounded-consumption HTTP stream with explicit response closure; no Parquet scan",
                "license_at_revision": getattr(info.card_data, "license", None),
                "selection": "first requested documents in each original upstream partition", "splits": {}}
    siblings = {entry.rfilename: entry for entry in info.siblings}
    for filename in SOURCE_FILES.values():
        if filename not in siblings:
            parser.error(f"Pinned revision has no original source file {filename}; inspect the dataset README instead of substituting another version")
    with tempfile.TemporaryDirectory(prefix="fm-download-", dir=out_dir) as temporary, httpx.Client(follow_redirects=True, timeout=60) as client:
        temporary = Path(temporary)
        for split, limit in (("train", args.max_train), ("validation", args.max_validation)):
            filename = SOURCE_FILES[split]
            url = hf_hub_url(DATASET, filename, repo_type="dataset", revision=revision)
            metadata["splits"][split] = download_partition(client, url, temporary / f"{split}.jsonl", limit, split)
            metadata["splits"][split].update({"source_file": filename, "source_url": url, "source_size_bytes": siblings[filename].size})
        (temporary / "source_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for filename in ("train.jsonl", "validation.jsonl", "source_metadata.json"):
            (temporary / filename).replace(out_dir / filename)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("Prepare preserving upstream validation: python scripts/prepare_data.py --input "
          f"{out_dir / 'train.jsonl'} --validation-input {out_dir / 'validation.jsonl'} --out-dir data/clean")


if __name__ == "__main__":
    main()
