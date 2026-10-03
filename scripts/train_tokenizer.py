"""Train educational BPE on train only, then stream packed document tokens."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from fm_tutorial.data import file_sha256, iter_jsonl_texts
from fm_tutorial.tokenizer import ByteBPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a toy byte BPE using train.jsonl only; tokenize train/val separately. Pair-count recomputation is slow on large corpora.")
    parser.add_argument("--data-dir", default="data/clean")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    try:
        for split in ("train", "val"):
            if not (data_dir / f"{split}.jsonl").is_file():
                raise ValueError(f"Missing {data_dir / (split + '.jsonl')}; run prepare_data.py first")
        tokenizer = ByteBPETokenizer.train(iter_jsonl_texts(data_dir / "train.jsonl"), args.vocab_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "token_dtype": "uint32", "byte_order": "little", "eos_id": tokenizer.EOS,
                    "pad_id": tokenizer.PAD, "vocab_size": tokenizer.vocab_size, "requested_vocab_size": args.vocab_size,
                    "tokenizer_training_split": "train", "document_separator": "one EOS after each document", "splits": {}}
        with tempfile.TemporaryDirectory(prefix="fm-tokenize-", dir=out_dir) as temporary:
            temporary = Path(temporary)
            tokenizer.save(temporary / "tokenizer.json")
            metadata["tokenizer_sha256"] = file_sha256(temporary / "tokenizer.json")
            for split in ("train", "val"):
                source = data_dir / f"{split}.jsonl"
                documents = tokens = 0
                with (temporary / f"{split}.bin").open("wb") as output:
                    for text in iter_jsonl_texts(source):
                        if not text:
                            raise ValueError(f"Empty document in {source}; prepare the data first")
                        encoded = np.asarray(tokenizer.encode(text, add_eos=True), dtype="<u4")
                        output.write(encoded.tobytes())
                        documents += 1
                        tokens += len(encoded)
                if not documents:
                    raise ValueError(f"{source} must contain a nonempty partition")
                metadata["splits"][split] = {"documents": documents, "tokens": tokens, "source": str(source),
                                            "source_sha256": file_sha256(source), "binary_sha256": file_sha256(temporary / f"{split}.bin")}
            report_path = data_dir / "data_report.json"
            if report_path.is_file():
                metadata["data_report"] = json.loads(report_path.read_text(encoding="utf-8"))
            (temporary / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for filename in ("tokenizer.json", "train.bin", "val.bin", "metadata.json"):
                (temporary / filename).replace(out_dir / filename)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
