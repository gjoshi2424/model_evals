"""Tests for utility functions in utils.py."""

import pytest

from utils import (
    count_consecutive_values,
    extract_before_parenthesis,
    extract_from_to,
    get_valid_name_city,
    is_valid_city_sequence,
    parse_json_plan,
    transportation_match,
)


def test_extract_before_parenthesis_with_state():
    assert extract_before_parenthesis("Chicago (IL)") == "Chicago "


def test_extract_before_parenthesis_no_parens():
    assert extract_before_parenthesis("Chicago") == "Chicago"


def test_extract_before_parenthesis_empty_string():
    assert extract_before_parenthesis("") == ""


def test_extract_before_parenthesis_multiple_parens():
    # Only the first parenthetical should be stripped
    result = extract_before_parenthesis("Salt Lake City (UT)(extra)")
    assert result == "Salt Lake City "


def test_get_valid_name_city_standard():
    name, city = get_valid_name_city("The Purple Pig, Chicago(IL)")
    assert name == "The Purple Pig"
    assert city == "Chicago"


def test_get_valid_name_city_no_state():
    name, city = get_valid_name_city("Chez Paul, Paris")
    assert name == "Chez Paul"
    assert city == "Paris"


def test_get_valid_name_city_unparseable():
    name, city = get_valid_name_city("no comma here")
    assert name == "-"
    assert city == "-"


def test_get_valid_name_city_strips_whitespace():
    name, city = get_valid_name_city("  Nobu ,  Malibu  ")
    assert name == "Nobu"
    assert city == "Malibu"


def test_count_consecutive_values_basic():
    result = count_consecutive_values(["A", "A", "B", "A"])
    assert result == [("A", 2), ("B", 1), ("A", 1)]


def test_count_consecutive_values_all_same():
    assert count_consecutive_values(["X", "X", "X"]) == [("X", 3)]


def test_count_consecutive_values_all_different():
    assert count_consecutive_values(["A", "B", "C"]) == [("A", 1), ("B", 1), ("C", 1)]


def test_count_consecutive_values_empty():
    assert count_consecutive_values([]) == []


def test_count_consecutive_values_single_element():
    assert count_consecutive_values(["A"]) == [("A", 1)]


def test_extract_from_to_standard():
    org, dest = extract_from_to("from New York to Los Angeles")
    assert org == "New York"
    assert dest == "Los Angeles"


def test_extract_from_to_no_match():
    org, dest = extract_from_to("no direction here")
    assert org is None
    assert dest is None


def test_extract_from_to_with_extra_text():
    org, dest = extract_from_to("Taxi from Chicago to Denver, 2 hours")
    assert org == "Chicago"
    assert dest == "Denver"


def test_extract_from_to_case_sensitive():
    # "From" with capital F should NOT match (pattern uses lowercase "from")
    org, dest = extract_from_to("From New York to Boston")
    assert org is None
    assert dest is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Taxi from NYC to LA, Cost: $200", "Taxi"),
        ("Self-driving from Dallas to Houston", "Self-driving"),
        ("Flight Number: AA123, from Boston to Miami", "Flight"),
        ("TAXI from A to B", "Taxi"),  # case-insensitive
        ("Walk to the park", None),
        ("", None),
    ],
)
def test_transportation_match(text, expected):
    assert transportation_match(text) == expected


def test_is_valid_city_sequence_valid_round_trip():
    # origin → dest1 (stays 2 days) → origin = valid
    assert is_valid_city_sequence(["NYC", "LA", "LA", "NYC"]) is True


def test_is_valid_city_sequence_multi_city_valid():
    # origin → dest1 (2 days) → dest2 (2 days) → origin = valid
    assert is_valid_city_sequence(["NYC", "LA", "LA", "SF", "SF", "NYC"]) is True


def test_is_valid_city_sequence_too_short():
    assert is_valid_city_sequence(["A", "B"]) is False


def test_is_valid_city_sequence_single_stay_in_middle_invalid():
    # LA visited once in the middle → invalid (must stay ≥ 2 nights)
    assert is_valid_city_sequence(["NYC", "LA", "NYC"]) is False


def test_is_valid_city_sequence_backtracking_invalid():
    # A → B → A → B pattern is invalid (B disappears then reappears)
    assert is_valid_city_sequence(["A", "B", "A", "B"]) is False


def test_is_valid_city_sequence_repeated_dest_valid():
    # Two destination cities each with 2 nights
    assert is_valid_city_sequence(["NYC", "LA", "LA", "CHI", "CHI", "NYC"]) is True


def test_parse_json_plan_code_block():
    raw = '```json\n[{"day": 1, "city": "NYC"}]\n```'
    result = parse_json_plan(raw)
    assert result == [{"day": 1, "city": "NYC"}]


def test_parse_json_plan_plain_code_block():
    raw = '```\n[{"x": 1}]\n```'
    result = parse_json_plan(raw)
    assert result == [{"x": 1}]


def test_parse_json_plan_bare_list():
    result = parse_json_plan('[{"x": 1}]')
    assert result == [{"x": 1}]


def test_parse_json_plan_multi_day():
    raw = '```json\n[{"day": 1}, {"day": 2}]\n```'
    result = parse_json_plan(raw)
    assert result is not None
    assert len(result) == 2
    assert result[0]["day"] == 1


def test_parse_json_plan_invalid_returns_none():
    assert parse_json_plan("not json at all !!!") is None


def test_parse_json_plan_empty_string_returns_none():
    assert parse_json_plan("") is None
