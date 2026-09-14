"""Per-character CTC confidence, and leaving out characters the model is unsure of."""

import pytest

from icinspect.ocr.charset import Charset

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-+./ "
CS = Charset(SYMBOLS)


def idx(ch: str) -> int:
    return SYMBOLS.index(ch) + 1


def test_collapse_keeps_repeats_that_a_blank_separates():
    frames = [idx("5"), idx("5"), 0, idx("5")]          # "55" then a blank then "5"
    assert CS.ctc_greedy_decode(frames) == "55"
    assert [r[0] for r in CS.collapse(frames)] == [idx("5"), idx("5")]


def test_scored_decode_spells_the_same_text():
    frames = [0, idx("N"), idx("N"), 0, idx("E"), 0, 0, idx("5")]
    scores = [0.9] * len(frames)
    scored = CS.ctc_greedy_decode_scored(frames, scores)
    assert "".join(ch for ch, _ in scored) == CS.ctc_greedy_decode(frames) == "NE5"


def test_a_character_scores_its_strongest_frame():
    """A wide glyph spans frames whose edges are weak; the peak is the commitment."""
    frames = [idx("W"), idx("W"), idx("W"), 0, idx("2")]
    scores = [0.40, 0.99, 0.55, 1.0, 0.70]
    assert CS.ctc_greedy_decode_scored(frames, scores) == [("W", 0.99), ("2", 0.70)]


def test_scored_decode_of_an_all_blank_read_is_empty():
    assert CS.ctc_greedy_decode_scored([0, 0, 0], [1.0, 1.0, 1.0]) == []


# --- the filter itself, over a stub model ---------------------------------

torch = pytest.importorskip("torch", reason="torch required")


class FixedLogits(torch.nn.Module):
    """Returns one prepared (T, N, C) tensor, whatever it is handed."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, _):
        return self.logits


def logits_for(pairs: list[tuple[str, float]]):
    """One frame per (character, peak probability), blank-separated."""
    frames = []
    for ch, p in pairs:
        row = torch.full((CS.num_classes,), (1.0 - p) / (CS.num_classes - 1))
        row[idx(ch)] = p
        frames.append(row)
        blank = torch.full((CS.num_classes,), 0.001)
        blank[0] = 0.999
        frames.append(blank)
    return torch.stack(frames).unsqueeze(1).log()      # (T, 1, C) as log-probs


def decode(pairs, floor):
    from icinspect.ocr.infer import _decode_batch

    model = FixedLogits(logits_for(pairs))
    return _decode_batch(model, CS, [torch.zeros(1, 32, 160)], torch.device("cpu"), floor)[0]


def test_unsure_characters_are_left_out():
    read = decode([("7", 0.35), ("9", 0.99), ("3", 0.99), ("M", 0.99)], floor=0.95)
    assert read["text"] == "93M"
    assert read["raw_text"] == "793M"
    assert read["dropped"] == 1


def test_zero_floor_keeps_everything():
    read = decode([("7", 0.35), ("9", 0.99)], floor=0.0)
    assert read["text"] == read["raw_text"] == "79"
    assert read["dropped"] == 0


def test_confidence_is_the_mean_over_reported_characters():
    read = decode([("7", 0.35), ("9", 0.99), ("3", 0.99)], floor=0.95)
    assert read["text"] == "93"
    assert read["confidence"] == pytest.approx(0.99, abs=0.01)   # the dropped 0.35 is not in it


def test_a_fully_rejected_read_is_empty_with_zero_confidence():
    read = decode([("7", 0.35), ("Q", 0.40)], floor=0.95)
    assert read["text"] == ""
    assert read["confidence"] == 0.0
    assert read["dropped"] == 2


# --- ignoring a whole mark ------------------------------------------------

def test_is_unreadable_rejects_a_mark_that_is_mostly_guesswork():
    """A logo decodes to a couple of characters, one confident by luck."""
    from icinspect.ocr.infer import is_unreadable

    logo = {"text": "7", "raw_text": "R7", "dropped": 1}      # half of it was unsure
    assert is_unreadable(logo, max_unsure=0.5)
    assert not is_unreadable(logo, max_unsure=1.0)            # rule disabled


def test_is_unreadable_keeps_a_worn_but_mostly_read_marking():
    from icinspect.ocr.infer import is_unreadable

    worn = {"text": "NE55P", "raw_text": "NE555P", "dropped": 1}
    assert not is_unreadable(worn, max_unsure=0.5)


def test_is_unreadable_rejects_a_segment_nothing_survived():
    from icinspect.ocr.infer import is_unreadable

    assert is_unreadable({"text": "", "raw_text": "QQ", "dropped": 2}, max_unsure=1.0)


def test_a_blank_segment_is_not_called_unreadable():
    """Nothing was decoded, so there is nothing to have got wrong."""
    from icinspect.ocr.infer import is_unreadable

    assert not is_unreadable({"text": "", "raw_text": "", "dropped": 0}, max_unsure=0.5)


def test_decode_settings_default_to_off():
    from icinspect.ocr.infer import max_unsure_frac, min_char_confidence

    assert min_char_confidence({}) == 0.0
    assert max_unsure_frac({}) == 1.0
    assert min_char_confidence({"decode": {"min_char_confidence": 0.9}}) == 0.9
    assert max_unsure_frac({"decode": {"max_unsure_frac": 0.5}}) == 0.5
