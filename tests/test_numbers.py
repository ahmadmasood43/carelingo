import pytest

from carelingo.numbers import contains_number, extract_numbers, unsupported_dates


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("Temperatur 36,9 °C, kein Fieber.", "de", [36.9]),
        ("Temperature 36.9, so no fever.", "en", [36.9]),
        ("Temperatura 36,9, czyli bez gorączki.", "pl", [36.9]),
        ("RR 145/90 mmHg, Puls 88/min.", "de", [145, 90, 88]),
        ("Pulse 92, sorry, 82.", "en", [92, 82]),
        ("Trinkmenge 1.500 ml", "de", [1500]),
        ("Fluid intake 1,500 ml", "en", [1500]),
        ("SpO2 91 %", "de", [91]),
        ("Metamizol 500mg", "de", [500]),
        ("Um 23:30 Uhr", "de", [23, 30]),
    ],
)
def test_extract_numbers(text, language, expected):
    assert extract_numbers(text, language) == expected


def test_decimal_separator_depends_on_language():
    # "47,5" is one number in German but two in an English sentence.
    assert extract_numbers("Metoprolol 47,5 mg", "de") == [47.5]
    assert not contains_number("Metoprolol 47,5 mg", "en", 47.5)


def test_invented_dates_are_found_even_though_numbers_after_dots_are_skipped():
    note = "Pflegebericht – 12.06.2024, Fr. Brandt, Zi. 12"
    assert unsupported_dates("Mrs Brandt, room 12.", note) == ("12.06.2024",)
    assert unsupported_dates("Mrs Brandt, room 12.", "Fr. Brandt, Zi. 12") == ()
