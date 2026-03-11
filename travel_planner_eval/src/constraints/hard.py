import math
import re
from typing import Any

import database
from utils import (
    extract_before_parenthesis,
    extract_from_to,
    get_valid_name_city,
)

def is_valid_transportation(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool | None, str | None]:
    """Check that the transportation mode satisfies the user's constraint.

    Args:
        question: Query metadata dict with fields: days, local_constraint.
        tested_data: List of per-day plan dicts with key: transportation.

    Returns:
        Tuple of ``(result, reason)`` where result is True (passes), False (fails), or
        None (constraint not set), and reason is a human-readable failure message or None.
    """
    if question["local_constraint"]["transportation"] is None:
        return None, None

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            if (
                question["local_constraint"]["transportation"] == "no flight"
                and "Flight" in value
            ):
                return (
                    False,
                    f"The transportation should not be {question['local_constraint']['transportation']}.",
                )
            elif (
                question["local_constraint"]["transportation"] == "no self-driving"
                and "Self-driving" in value
            ):
                return (
                    False,
                    f"The transportation should not be {question['local_constraint']['transportation']}.",
                )

    return True, None


def get_total_cost(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> float:
    """Compute the total monetary cost of a plan.

    Args:
        question: Query metadata dict with fields: days, people_number.
        tested_data: List of per-day plan dicts with keys: transportation, current_city,
            breakfast, lunch, dinner, accommodation.

    Returns:
        Total cost as a float.
    """
    total_cost = 0.0

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        # Transportation
        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit["current_city"])

            if org_city is not None and dest_city is not None:
                if "flight number" in value.lower():
                    flight_num = value.split("Flight Number: ")[1].split(",")[0]
                    df = database.flights()
                    res = df[df["Flight Number"] == flight_num]
                    if len(res) > 0:
                        total_cost += res["Price"].values[0] * question["people_number"]

                elif "self-driving" in value.lower() or "taxi" in value.lower():
                    mode = "self-driving" if "self-driving" in value.lower() else "taxi"
                    cost = database.distance_cost(
                        extract_before_parenthesis(org_city),
                        extract_before_parenthesis(dest_city),
                        mode,
                    )
                    if cost is not None:
                        if mode == "self-driving":
                            total_cost += cost * math.ceil(
                                question["people_number"] / 5
                            )
                        else:
                            total_cost += cost * math.ceil(
                                question["people_number"] / 4
                            )

        # Meals
        for meal in ("breakfast", "lunch", "dinner"):
            if unit.get(meal) and unit[meal] != "-":
                name, city = get_valid_name_city(unit[meal])
                df = database.restaurants()
                res = df[
                    (df["Name"].astype(str).str.contains(re.escape(name)))
                    & (df["City"] == city)
                ]
                if len(res) > 0:
                    total_cost += (
                        res["Average Cost"].values[0] * question["people_number"]
                    )

        # Accommodation
        if unit.get("accommodation") and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            df = database.accommodations()
            res = df[
                (df["NAME"].astype(str).str.contains(re.escape(name)))
                & (df["city"] == city)
            ]
            if len(res) > 0:
                total_cost += res["price"].values[0] * math.ceil(
                    question["people_number"] / res["maximum occupancy"].values[0]
                )

    return total_cost


def is_valid_room_rule(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool | None, str | None]:
    """Check that each accommodation's house rules satisfy the user's constraint.

    Args:
        question: Query metadata dict with fields: days, local_constraint.
        tested_data: List of per-day plan dicts with key: accommodation.

    Returns:
        Tuple of ``(result, reason)`` where result is True (passes), False (fails), or
        None (constraint not set), and reason is a human-readable failure message or None.
    """
    if question["local_constraint"]["house rule"] is None:
        return None, None

    rule = question["local_constraint"]["house rule"]
    rule_map = {
        "smoking": "No smoking",
        "parties": "No parties",
        "children under 10": "No children under 10",
        "visitors": "No visitors",
        "pets": "No pets",
    }
    forbidden_text = rule_map.get(rule)

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            df = database.accommodations()
            res = df[
                (df["NAME"].astype(str).str.contains(re.escape(name)))
                & (df["city"] == city)
            ]
            if len(res) > 0 and forbidden_text is not None:
                if forbidden_text in str(res["house_rules"].values[0]):
                    return False, f"The house rule should be {rule}."

    return True, None


def is_valid_cuisine(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool | None, str | None]:
    """Check that all requested cuisine types appear at least once across the trip.

    Args:
        question: Query metadata dict with fields: days, org, local_constraint.
        tested_data: List of per-day plan dicts with keys: breakfast, lunch, dinner.

    Returns:
        Tuple of ``(result, reason)`` where result is True (passes), False (fails), or
        None (constraint not set), and reason is a human-readable failure message or None.
    """
    if not question["local_constraint"]["cuisine"]:
        return None, None

    cuisine_set: set[str] = set()
    df = database.restaurants()

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        for meal in ("breakfast", "lunch", "dinner"):
            if unit.get(meal) and unit[meal] != "-":
                name, city = get_valid_name_city(unit[meal])
                if city == question["org"]:
                    continue
                res = df[
                    (df["Name"].astype(str).str.contains(re.escape(name)))
                    & (df["City"] == city)
                ]
                if len(res) > 0:
                    for cuisine in question["local_constraint"]["cuisine"]:
                        if cuisine in res.iloc[0]["Cuisines"]:
                            cuisine_set.add(cuisine)

    for cuisine in question["local_constraint"]["cuisine"]:
        if cuisine not in cuisine_set:
            return False, f"The cuisine {cuisine} is not satisfied."

    return True, None


def is_valid_room_type(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool | None, str | None]:
    """Check that each accommodation matches the required room type.

    Args:
        question: Query metadata dict with fields: days, local_constraint.
        tested_data: List of per-day plan dicts with key: accommodation.

    Returns:
        Tuple of ``(result, reason)`` where result is True (passes), False (fails), or
        None (constraint not set), and reason is a human-readable failure message or None.
    """
    if question["local_constraint"]["room type"] is None:
        return None, None

    room_type = question["local_constraint"]["room type"]
    # Maps constraint value → (db_value, must_match)
    room_type_map: dict[str, tuple[str, bool]] = {
        "not shared room": ("Shared room", False),
        "shared room": ("Shared room", True),
        "private room": ("Private room", True),
        "entire room": ("Entire home/apt", True),
    }

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if unit.get("accommodation") and unit["accommodation"] != "-":
            name, city = get_valid_name_city(unit["accommodation"])
            df = database.accommodations()
            res = df[
                (df["NAME"].astype(str).str.contains(re.escape(name)))
                & (df["city"] == city)
            ]
            if len(res) > 0 and room_type in room_type_map:
                db_value, must_match = room_type_map[room_type]
                actual = res["room type"].values[0]
                if must_match and actual != db_value:
                    return False, f"The room type should be {room_type}."
                if not must_match and actual == db_value:
                    return False, f"The room type should be {room_type}."

    return True, None


# ---------------------------------------------------------------------------
# Aggregate evaluation entry point
# ---------------------------------------------------------------------------


def evaluation(
    query_data: dict[str, Any], tested_data: list[dict[str, Any]]
) -> dict[str, tuple]:
    """Run all hard constraint checks on a parsed plan.

    Args:
        query_data: Query metadata dict with fields: days, people_number, budget,
            local_constraint.
        tested_data: List of per-day plan dicts with keys: transportation, current_city,
            breakfast, lunch, dinner, accommodation.

    Returns:
        Dict mapping check name to ``(result, reason)`` tuples. result is True (passes),
        False (fails), or None (constraint not applicable), and reason is a
        human-readable failure message or None.
    """
    return_info: dict[str, tuple] = {}
    return_info["valid_transportation"] = is_valid_transportation(
        query_data, tested_data
    )
    return_info["valid_cuisine"] = is_valid_cuisine(query_data, tested_data)
    return_info["valid_room_rule"] = is_valid_room_rule(query_data, tested_data)
    return_info["valid_room_type"] = is_valid_room_type(query_data, tested_data)
    return_info["valid_cost"] = (
        bool(get_total_cost(query_data, tested_data) <= query_data["budget"]),
        None,
    )
    return return_info
