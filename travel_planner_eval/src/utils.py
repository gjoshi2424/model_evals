import json
import re
from typing import Any
import ast


def extract_before_parenthesis(s: str) -> str:
    """Extract the portion of a string before any parenthetical expression.

    Args:
        s: Input string, potentially containing parenthetical content.

    Returns:
        String with everything from the first ``(`` onwards removed,
        or the original string unchanged if no parentheses are found.
    """
    match = re.search(r"^(.*?)\([^)]*\)", s)
    return match.group(1) if match else s


def get_valid_name_city(info: str) -> tuple[str, str]:
    """Parse 'Name, City(State)' style strings into (name, city) tuples.

    Args:
        info: String in the format ``"Name, City(State)"`` or ``"Name, City"``.

    Returns:
        Tuple of ``(name, city)`` with any state suffix stripped from city.
        Returns ``("-", "-")`` if the string cannot be parsed.
    """
    pattern = r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$"
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(
            match.group(2).strip()
        ).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-", "-"


def count_consecutive_values(lst: list) -> list[tuple]:
    """Group consecutive identical values in a list.
    Args:
        lst: List of comparable values.

    Returns:
        List of ``(value, count)`` tuples for each run of identical values.
        Returns an empty list if ``lst`` is empty.
    """
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))
    return result


def extract_from_to(text: str) -> tuple[str | None, str | None]:
    """Extract origin and destination from 'from A to B' style strings.

    Args:
        text: String that may contain a ``"from X to Y"`` pattern.

    Returns:
        Tuple of ``(origin, destination)`` strings, or ``(None, None)`` if
        no match is found.
    """
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


def transportation_match(text: str) -> str | None:
    """Classify a transportation description string into a canonical mode.

    Args:
        text: Free-form transportation description (e.g. from a plan day entry).

    Returns:
        One of ``"Taxi"``, ``"Self-driving"``, or ``"Flight"``, or None if the
        mode cannot be determined.
    """
    if "taxi" in text.lower():
        return "Taxi"
    elif "self-driving" in text.lower():
        return "Self-driving"
    elif "flight" in text.lower():
        return "Flight"
    return None


def is_valid_city_sequence(city_list: list[str]) -> bool:
    """Check that a city visit sequence is valid.

    Args:
        city_list: Ordered list of city names representing the day-by-day itinerary.

    Returns:
        True if the sequence is valid, False otherwise.
    """
    min_cities = 3  # origin + at least one destination + return to origin
    if len(city_list) < min_cities:
        return False

    visited_cities: set[str] = set()
    i = 0
    while i < len(city_list):
        city = city_list[i]

        if city in visited_cities and (i != 0 and i != len(city_list) - 1):
            return False

        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1

        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False

        visited_cities.add(city)

    return True


def parse_json_plan(raw: str) -> list[dict[str, Any]] | None:
    """Extract and parse a JSON plan from LLM output.

    Args:
        raw: Raw LLM output string, expected to contain a JSON array of day-plan dicts.

    Returns:
        Parsed list of per-day plan dicts, or None if all parsing attempts fail.
    """
    # Try ```json ... ``` code block first (standard GPT-4 format)
    try:
        json_str = raw.split("```json")[1].split("```")[0].strip()
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError):
        pass

    # Try bare ``` ... ``` code block
    try:
        json_str = raw.split("```")[1].split("```")[0].strip()
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError):
        pass
    try:
        result = ast.literal_eval(raw.strip())
        if isinstance(result, list):
            return result
    except Exception:
        pass

    return None
