"""Small text fixtures are program diagnostics, not a training dataset."""
import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_tutorial.data import prepare_jsonl
from fm_tutorial.tokenizer import ByteBPETokenizer


ROOT = Path(__file__).resolve().parents[1]


def download_module():
    spec = importlib.util.spec_from_file_location("tutorial_download_tinystories", ROOT / "scripts/download_tinystories.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, texts):
    path.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts), encoding="utf-8")


def read_texts(path):
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


class DataTests(unittest.TestCase):
    def test_original_text_parser_preserves_unicode_paragraphs_and_rejects_truncation(self):
        module = download_module()
        self.assertTrue(hasattr(module, "iter_original_documents"), "original-text parser is missing")
        raw = "第一段🙂\n第二段\n<|endoftext|>\n另一个\n<|endoftext|>\n".encode("utf-8")
        chunks = [raw[index:index + 1] for index in range(len(raw))]
        self.assertEqual(list(module.iter_original_documents(chunks)), ["第一段🙂\n第二段", "另一个"])
        with self.assertRaisesRegex(ValueError, "unterminated|truncated"):
            list(module.iter_original_documents([b"partial story"]))
        with self.assertRaises(UnicodeDecodeError):
            list(module.iter_original_documents([b"\xff\n<|endoftext|>\n"]))

    def test_downloader_partition_writer_closes_early_iterator_and_keeps_partitions(self):
        module = download_module()
        self.assertTrue(hasattr(module, "write_partition"), "bounded partition writer is missing")
        closed = []
        consumed = []

        def documents():
            try:
                for text in ("training diagnostic", "must not be consumed"):
                    consumed.append(text)
                    yield text
            finally:
                closed.append(True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = module.write_partition(documents(), root / "train.jsonl", 1, "train")
            module.write_partition(iter(["validation diagnostic"]), root / "validation.jsonl", 1, "validation")
            self.assertEqual(read_texts(root / "train.jsonl"), ["training diagnostic"])
            self.assertEqual(read_texts(root / "validation.jsonl"), ["validation diagnostic"])
            self.assertEqual(consumed, ["training diagnostic"])
            self.assertEqual(closed, [True])
            self.assertEqual(report["documents"], 1)

    def test_normalize_deduplicate_before_stable_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            texts = [f"diagnostic document {index}\nsecond line" for index in range(40)]
            texts += [" café\r\nline ", "cafe\u0301\nline", "  ", "x"]
            write_jsonl(root / "raw.jsonl", texts)
            report = prepare_jsonl(root / "raw.jsonl", root / "a", min_chars=4, val_fraction=0.25)
            first_train = read_texts(root / "a/train.jsonl")
            first_val = read_texts(root / "a/val.jsonl")
            self.assertTrue(first_train)
            self.assertTrue(first_val)
            self.assertFalse(set(first_train) & set(first_val))
            self.assertEqual(len(first_train) + len(first_val), 41)
            self.assertIn("café\nline", first_train + first_val)
            self.assertEqual(report["duplicates"], 1)
            self.assertEqual(report["rejected_short"], 2)
            write_jsonl(root / "raw.jsonl", list(reversed(texts)))
            prepare_jsonl(root / "raw.jsonl", root / "b", min_chars=4, val_fraction=0.25)
            self.assertEqual(set(first_train), set(read_texts(root / "b/train.jsonl")))
            self.assertEqual(set(first_val), set(read_texts(root / "b/val.jsonl")))
            self.assertTrue((root / "a/data_report.json").exists())

    def test_explicit_validation_retains_partition_and_removes_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "train.jsonl", ["café", "training document"])
            write_jsonl(root / "validation.jsonl", ["cafe\u0301", "validation document", "validation document"])
            report = prepare_jsonl(root / "train.jsonl", root / "clean", min_chars=2, validation_input=root / "validation.jsonl")
            self.assertEqual(read_texts(root / "clean/train.jsonl"), ["café", "training document"])
            self.assertEqual(read_texts(root / "clean/val.jsonl"), ["validation document"])
            self.assertEqual(report["validation_overlap_removed"], 1)

    def test_tiny_invalid_and_bad_rows_fail_with_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "raw.jsonl", ["one"])
            with self.assertRaisesRegex(ValueError, "nonempty|more|documents"):
                prepare_jsonl(root / "raw.jsonl", root / "clean", min_chars=1)
            root.joinpath("raw.jsonl").write_text('{"text": 42}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                prepare_jsonl(root / "raw.jsonl", root / "clean", min_chars=1)
            with self.assertRaises(ValueError):
                prepare_jsonl(root / "raw.jsonl", root / "clean", val_fraction=1)

    def test_tokenizer_cli_only_learns_train_and_writes_document_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("clean").mkdir()
            write_jsonl(root / "clean/train.jsonl", ["aaaa", "bbbb"])
            write_jsonl(root / "clean/val.jsonl", ["zzzz"])
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_tokenizer.py"), "--data-dir", str(root / "clean"), "--out-dir", str(root / "processed"), "--vocab-size", "262"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            tokenizer = ByteBPETokenizer.load(root / "processed/tokenizer.json")
            self.assertEqual(tokenizer.encode("zzzz"), [122, 122, 122, 122])
            train = np.fromfile(root / "processed/train.bin", dtype="<u4")
            validation = np.fromfile(root / "processed/val.bin", dtype="<u4")
            self.assertEqual(int((train == 256).sum()), 2)
            self.assertEqual(validation.tolist(), [122, 122, 122, 122, 256])
            self.assertTrue((root / "processed/metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
