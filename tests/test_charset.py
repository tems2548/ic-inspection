"""Charset / CTC coding tests. No torch needed."""

import pytest

from icinspect.ocr.charset import BLANK, Charset, cer, levenshtein

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-+./ "


@pytest.fixture
def charset() -> Charset:
    return Charset(symbols=SYMBOLS, uppercase=True)


def test_blank_is_reserved(charset):
    assert BLANK == 0
    assert charset.num_classes == len(SYMBOLS) + 1
    assert all(i > 0 for i in charset.encode("STM32"))


def test_roundtrip_uppercases(charset):
    assert charset.decode_indices(charset.encode("stm32f407vgt6")) == "STM32F407VGT6"


def test_rejects_unknown_characters(charset):
    assert not charset.is_encodable("STM32_F4")
    assert charset.unknown_chars("STM32_F4") == {"_"}
    with pytest.raises(ValueError, match="not in the charset"):
        charset.encode("STM32_F4")


def test_duplicate_symbols_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        Charset(symbols="AAB")


def test_greedy_decode_collapses_repeats_then_drops_blanks(charset):
    # A=11, B=12 in this charset (offset by 1 for blank).
    a, b = charset.encode("AB")
    # "A A blank A B B" -> "AAB": the blank separates the two A runs.
    assert charset.ctc_greedy_decode([a, a, BLANK, a, b, b]) == "AAB"
    # Without the blank, the repeat collapses to a single A.
    assert charset.ctc_greedy_decode([a, a, a, b]) == "AB"
    assert charset.ctc_greedy_decode([BLANK, BLANK]) == ""


def test_case_sensitive_charset_preserves_case():
    cs = Charset(symbols="abcABC", uppercase=False)
    assert cs.decode_indices(cs.encode("aBc")) == "aBc"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("abc", "", 3), ("", "abc", 3), ("ab", "ba", 2)],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


def test_cer_edges():
    assert cer("", "") == 0.0
    assert cer("X", "") == 1.0          # spurious read against an empty label
    assert cer("STM32F5", "STM32F4") == pytest.approx(1 / 7)
    assert cer("", "ABCD") == 1.0
