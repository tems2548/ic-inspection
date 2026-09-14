"""Character set and CTC encode/decode. Pure Python -- no torch import."""

from __future__ import annotations

from dataclasses import dataclass

BLANK = 0  # CTC blank is always index 0


@dataclass(frozen=True)
class Charset:
    symbols: str
    uppercase: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "Charset":
        c = cfg.get("charset", cfg)
        return cls(symbols=c["symbols"], uppercase=bool(c.get("uppercase", True)))

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            dupes = sorted({ch for ch in self.symbols if self.symbols.count(ch) > 1})
            raise ValueError(f"Duplicate characters in charset: {dupes!r}")

    @property
    def num_classes(self) -> int:
        """Symbols + 1 for the CTC blank."""
        return len(self.symbols) + 1

    def normalize(self, text: str) -> str:
        return text.upper() if self.uppercase else text

    def is_encodable(self, text: str) -> bool:
        return all(ch in self.symbols for ch in self.normalize(text))

    def unknown_chars(self, text: str) -> set[str]:
        return {ch for ch in self.normalize(text) if ch not in self.symbols}

    def encode(self, text: str) -> list[int]:
        """Text -> class indices (1-based; 0 is reserved for blank)."""
        text = self.normalize(text)
        missing = self.unknown_chars(text)
        if missing:
            raise ValueError(f"Characters {sorted(missing)!r} are not in the charset")
        return [self.symbols.index(ch) + 1 for ch in text]

    def decode_indices(self, indices: list[int]) -> str:
        """Class indices -> text, ignoring blanks. No collapsing."""
        return "".join(self.symbols[i - 1] for i in indices if i != BLANK)

    def collapse(self, frame_indices: list[int]) -> list[tuple[int, int, int]]:
        """CTC best-path collapse -> [(class index, first frame, last frame)].

        Repeats collapse into one character unless a blank separates them, which
        is what lets the model spell "555".
        """
        runs: list[tuple[int, int, int]] = []
        prev = None
        for i, idx in enumerate(frame_indices):
            if idx != BLANK:
                if idx != prev:
                    runs.append((idx, i, i))
                else:
                    runs[-1] = (idx, runs[-1][1], i)
            prev = idx
        return runs

    def ctc_greedy_decode(self, frame_indices: list[int]) -> str:
        """Best-path decode: collapse repeats, then drop blanks."""
        return "".join(self.symbols[idx - 1] for idx, _, _ in self.collapse(frame_indices))

    def ctc_greedy_decode_scored(
        self, frame_indices: list[int], frame_scores: list[float]
    ) -> list[tuple[str, float]]:
        """Best-path decode keeping a confidence per emitted character.

        A character's score is its **strongest** frame, not the mean over its
        run: CTC output is peaky, and a wide glyph spreads over frames whose
        edges are half-way into the neighbouring character. Averaging those in
        penalises wide letters for being wide, which is not what "unsure" means.
        """
        return [
            (self.symbols[idx - 1], max(frame_scores[first : last + 1]))
            for idx, first, last in self.collapse(frame_indices)
        ]


def levenshtein(a: str, b: str) -> int:
    """Edit distance, used for CER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, target: str) -> float:
    """Character error rate; 0.0 when both are empty."""
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)
