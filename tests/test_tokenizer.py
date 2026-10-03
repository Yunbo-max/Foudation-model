"""Tiny diagnostic strings test algorithms, not model quality or benchmarks."""
import json
import tempfile
import unittest
from pathlib import Path

from fm_tutorial.tokenizer import ByteBPETokenizer


class TokenizerTests(unittest.TestCase):
    def test_roundtrip_unseen_unicode_and_document_eos(self):
        tokenizer = ByteBPETokenizer.train(["banana banana", "你好，世界"], 270)
        for text in ["", "🙂 café\n你好\t𐍈", "banana", "\x00"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.encode(text, add_eos=True)[-1], 256)
            self.assertEqual(tokenizer.decode(tokenizer.encode(text) + [256, 257]), text)

    def test_merge_order_is_deterministic_and_compresses_repeated_bytes(self):
        first = ByteBPETokenizer.train(["abab abab", "xyxy"], 264)
        second = ByteBPETokenizer.train(["xyxy", "abab abab"], 264)
        self.assertEqual(first.encode("abab abab"), second.encode("abab abab"))
        self.assertLess(len(first.encode("abab abab")), len("abab abab".encode()))
        self.assertLessEqual(first.vocab_size, 264)

    def test_merges_do_not_cross_document_boundaries(self):
        tokenizer = ByteBPETokenizer.train(["a", "b"], 270)
        self.assertEqual(tokenizer.vocab_size, 258)
        self.assertEqual(tokenizer.encode("ab"), [97, 98])

    def test_persistence_keeps_token_identity(self):
        tokenizer = ByteBPETokenizer.train(["hello hello 中文"], 270)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)
            self.assertEqual(restored.vocab_size, tokenizer.vocab_size)
            self.assertEqual(restored.encode("hello 中文"), tokenizer.encode("hello 中文"))

    def test_invalid_vocab_and_corrupted_merges_rejected(self):
        for size in [257, 259.5, True]:
            with self.assertRaises(ValueError):
                ByteBPETokenizer.train(["hello"], size)
        with self.assertRaises(ValueError):
            ByteBPETokenizer.train([], 260)
        tokenizer = ByteBPETokenizer.train(["abab"], 260)
        with self.assertRaises(ValueError):
            tokenizer.decode([tokenizer.vocab_size])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            payload = json.loads(path.read_text())
            payload["merges"][0] = [99999, 97]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ByteBPETokenizer.load(path)

    def test_file_rejects_inconsistent_or_noninteger_header_ids(self):
        tokenizer = ByteBPETokenizer.train(["abab"], 260)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            payload = json.loads(path.read_text())
            corruptions = [
                {"vocab_size": 999},
                {"vocab_size": float(payload["vocab_size"])},
                {"version": True},
                {"special_tokens": {"eos": 256.0, "pad": 257}},
            ]
            for corruption in corruptions:
                with self.subTest(corruption=corruption):
                    path.write_text(json.dumps(dict(payload, **corruption)))
                    with self.assertRaises(ValueError):
                        ByteBPETokenizer.load(path)


if __name__ == "__main__":
    unittest.main()
