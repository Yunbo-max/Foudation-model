"""Original, deliberately small byte-level BPE for teaching.

Training recomputes pair counts after each merge: transparent but much slower
than production tokenizers. Documents remain separate throughout training.
"""
from collections import Counter
import json
from numbers import Integral
from pathlib import Path


class ByteBPETokenizer:
    EOS = 256
    PAD = 257
    BASE_VOCAB_SIZE = 258

    def __init__(self, merges=()):
        self.merges = []
        self._pieces = {index: bytes([index]) for index in range(256)}
        self._ranks = {}
        for rank, pair in enumerate(merges):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each BPE merge must contain exactly two token IDs")
            if any(not isinstance(token, Integral) or isinstance(token, bool) for token in pair):
                raise ValueError("BPE merge IDs must be integers")
            pair = tuple(int(token) for token in pair)
            if pair in self._ranks or any(token not in self._pieces for token in pair):
                raise ValueError("Invalid merge: duplicate pair, special token, or forward reference")
            token_id = self.BASE_VOCAB_SIZE + rank
            self.merges.append(pair)
            self._ranks[pair] = (rank, token_id)
            self._pieces[token_id] = self._pieces[pair[0]] + self._pieces[pair[1]]

    @property
    def vocab_size(self):
        return self.BASE_VOCAB_SIZE + len(self.merges)

    @staticmethod
    def _merge(sequence, pair, new_id):
        result = []
        cursor = 0
        while cursor < len(sequence):
            if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                result.append(new_id)
                cursor += 2
            else:
                result.append(sequence[cursor])
                cursor += 1
        return tuple(result)

    @classmethod
    def train(cls, texts, vocab_size):
        """Learn at most ``vocab_size`` entries, using training documents only.

        Ties are broken by the numeric token pair, independent of corpus order.
        The final vocabulary can be smaller when no adjacent pair remains.
        """
        if not isinstance(vocab_size, Integral) or isinstance(vocab_size, bool) or vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError("vocab_size must be an integer at least 258")
        sequences = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Tokenizer training documents must be strings")
            if text:
                sequences[tuple(text.encode("utf-8"))] += 1
        if not sequences:
            raise ValueError("Tokenizer needs at least one nonempty training document")
        merges = []
        while cls.BASE_VOCAB_SIZE + len(merges) < vocab_size:
            pairs = Counter()
            for sequence, weight in sequences.items():
                for pair in zip(sequence, sequence[1:]):
                    pairs[pair] += weight
            if not pairs:
                break
            chosen = min(pairs, key=lambda pair: (-pairs[pair], pair))
            new_id = cls.BASE_VOCAB_SIZE + len(merges)
            replacement = Counter()
            for sequence, weight in sequences.items():
                replacement[cls._merge(sequence, chosen, new_id)] += weight
            sequences = replacement
            merges.append(chosen)
        return cls(merges)

    def encode(self, text, add_eos=False):
        if not isinstance(text, str):
            raise ValueError("encode expects a Unicode string")
        sequence = tuple(text.encode("utf-8"))
        while len(sequence) > 1:
            candidates = [(self._ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self._ranks]
            if not candidates:
                break
            (_, new_id), pair = min(candidates)
            sequence = self._merge(sequence, pair, new_id)
        result = list(sequence)
        if add_eos:
            result.append(self.EOS)
        return result

    def decode(self, ids):
        """Decode bytes; skip EOS/PAD and replace incomplete generated UTF-8.

        Any sequence produced by ``encode`` round-trips exactly. Arbitrarily
        sampled byte IDs may be incomplete UTF-8 and use the replacement glyph.
        """
        pieces = []
        for token in ids:
            if not isinstance(token, Integral) or isinstance(token, bool) or not 0 <= token < self.vocab_size:
                raise ValueError(f"Unknown token ID: {token!r}")
            if token not in (self.EOS, self.PAD):
                pieces.append(self._pieces[int(token)])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "fm-tutorial-byte-bpe", "version": 1,
                   "special_tokens": {"eos": self.EOS, "pad": self.PAD},
                   "vocab_size": self.vocab_size, "merges": self.merges}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("format") != "fm-tutorial-byte-bpe" or type(payload.get("version")) is not int or payload.get("version") != 1:
                raise ValueError("Unsupported tokenizer format/version")
            special = payload.get("special_tokens")
            if not isinstance(special, dict) or any(type(special.get(key)) is not int for key in ("eos", "pad")) or special != {"eos": cls.EOS, "pad": cls.PAD}:
                raise ValueError("Tokenizer special token IDs disagree with the file format")
            if not isinstance(payload.get("merges"), list):
                raise ValueError("Tokenizer merges must be a list")
            tokenizer = cls(payload["merges"])
            if type(payload.get("vocab_size")) is not int or payload.get("vocab_size") != tokenizer.vocab_size:
                raise ValueError("Tokenizer vocab_size does not match its merges")
            return tokenizer
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            raise ValueError(f"Invalid tokenizer file: {error}") from error
