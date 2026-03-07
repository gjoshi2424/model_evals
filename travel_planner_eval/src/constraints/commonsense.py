"""
Commonsense constraint evaluation for TravelPlanner.

Ported from the original TravelPlanner repository (evaluation/commonsense_constraint.py).

All checks from the original are implemented, including those that require the bundled
travel database CSV files.
"""

import re
from typing import Any

import database
from utils import (
    count_consecutive_values,
    extract_before_parenthesis,
    extract_from_to,
    get_valid_name_city,
    is_valid_city_sequence,
    transportation_match,
)

# Days threshold above which cities must belong to the destination state (original uses 3)
_MIN_DAYS_FOR_STATE_CHECK = 3

# ---------------------------------------------------------------------------
# Individual constraint checks
# Copied / adapted from evaluation/commonsense_constraint.py
# ---------------------------------------------------------------------------


def is_reasonable_visiting_city(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that the city visit sequence forms a valid closed-loop itinerary.

    Verifies:
    - First city is the departure city (org)
    - Trip is a closed circle (returns to departure)
    - City sequence has no invalid back-and-forth patterns
    - All cities are valid and within the destination state (for trips > 3 days)

    Adapted from evaluation/commonsense_constraint.py.
    """
    city_list = []

    for i in range(min(question["days"], len(tested_data))):
        city_value = tested_data[i]["current_city"]

        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question["org"]:
                return False, f"The first day's city should be {question['org']}."
            city_list += [city1, city2]
        else:
            city_list.append(extract_before_parenthesis(city_value))

    if city_list[0] != city_list[-1]:
        return False, "The trip should be a closed circle."

    if not is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."

    city_state = database.city_state_map()
    for idx, city in enumerate(city_list):
        if city not in city_state:
            return False, f"{city} is not a valid city."
        if (
            idx not in [0, len(city_list) - 1]
            and question["days"] > _MIN_DAYS_FOR_STATE_CHECK
            and city_state[city] != question["dest"]
        ):
            return False, f"{city} is not in {question['dest']}."

    return True, None


def is_valid_restaurants(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that no restaurant is visited more than once across the trip.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    restaurants_list: list[str] = []

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            if unit["breakfast"] not in restaurants_list:
                restaurants_list.append(unit["breakfast"])
            else:
                return False, f"The restaurant in day {i + 1} breakfast is repeated."

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            if unit["lunch"] not in restaurants_list:
                restaurants_list.append(unit["lunch"])
            else:
                return (
                    False,
                    f"The restaurant in day {i + 1} lunch {unit['lunch']} is repeated.",
                )

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            if unit["dinner"] not in restaurants_list:
                restaurants_list.append(unit["dinner"])
            else:
                return False, f"The restaurant in day {i + 1} dinner is repeated."

    return True, None


def is_valid_attractions(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that no attraction is visited more than once across the trip.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    attractions_list: list[str] = []

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            for attraction in unit["attraction"].split(";")[:-1]:
                if attraction not in attractions_list:
                    attractions_list.append(attraction)
                else:
                    return (
                        False,
                        f"The attraction '{attraction}' in day {i + 1} is repeated.",
                    )

    return True, None


def is_valid_transportation(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that no conflicting transportation modes are used (e.g. flight + self-driving).

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    if tested_data[0]["transportation"] and tested_data[0]["transportation"] != "-":
        transportation_list = [transportation_match(tested_data[0]["transportation"])]
    else:
        return False, "The transportation in day 1 should not be empty."

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if (
            "transportation" in unit
            and unit["transportation"]
            and unit["transportation"] != "-"
        ):
            transportation_list.append(transportation_match(unit["transportation"]))

    if ("Self-driving" in transportation_list and "Flight" in transportation_list) or (
        "Taxi" in transportation_list and "Self-driving" in transportation_list
    ):
        return False, "The transportation is conflicting."

    return True, None


def is_valid_information_in_current_city(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that each day's meals, attractions, and accommodation match the current city.

    Ensures each day entry's meals, attractions, and accommodation refer to a city
    consistent with the day's current city.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        current_city = unit["current_city"]
        final_city_list: list[str] = []

        if "from" in current_city:
            city1, city2 = extract_from_to(current_city)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            final_city_list = [city1, city2]
        else:
            final_city_list = [extract_before_parenthesis(current_city)]

        if (
            "transportation" in unit
            and unit["transportation"]
            and unit["transportation"] != "-"
        ):
            for city in final_city_list:
                if city not in unit["transportation"]:
                    return (
                        False,
                        f"The transportation in day {i + 1} is invalid city choice.",
                    )

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            flag = any(city in unit["breakfast"] for city in final_city_list)
            if not flag:
                return (
                    False,
                    f"The breakfast in day {i + 1} is invalid city choice.",
                )

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            flag = any(city in unit["lunch"] for city in final_city_list)
            if not flag:
                return False, f"The lunch in day {i + 1} is invalid city choice."

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            flag = any(city in unit["dinner"] for city in final_city_list)
            if not flag:
                return False, f"The dinner in day {i + 1} is invalid city choice."

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            attraction_list = unit["attraction"].split(";")[:-1]
            for attraction in attraction_list:
                flag = any(city in attraction for city in final_city_list)
                if not flag:
                    return (
                        False,
                        f"The attraction in day {i + 1} is invalid city choice.",
                    )

        if (
            "accommodation" in unit
            and unit["accommodation"]
            and unit["accommodation"] != "-"
        ):
            if final_city_list[-1] not in unit["accommodation"]:
                return (
                    False,
                    f"The accommodation in day {i + 1} is invalid city choice.",
                )

    return True, None


def is_valid_information_in_sandbox(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that all named entities in the plan exist in the database.

    Verifies flight numbers, restaurants, attractions, and accommodations against
    the bundled database. Known as the "sandbox" validation because it checks
    grounding against the set of valid entities available to the solver.
    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if unit["transportation"] and unit["transportation"] != "-":
            value = unit["transportation"]
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit["current_city"])

            if "flight number" in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError as err:
                    raise ValueError(
                        f"The transportation {value} in day {i + 1} can not be parsed."
                    ) from err
                flight_num = value.split("Flight Number: ")[1].split(",")[0]
                df = database.flights()
                if (
                    len(
                        df[
                            (df["Flight Number"] == flight_num)
                            & (df["OriginCityName"] == org_city)
                            & (df["DestCityName"] == dest_city)
                        ]
                    )
                    < 1
                ):
                    return (
                        False,
                        f"The flight number in day {i + 1} is invalid in the sandbox.",
                    )

            elif "self-driving" in value.lower() or "taxi" in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city = "-"
                    dest_city = "-"
                mode = "self-driving" if "self-driving" in value.lower() else "taxi"
                if database.distance_cost(org_city, dest_city, mode) is None:
                    return (
                        False,
                        f"The {mode} in day {i + 1} is invalid in the sandbox.",
                    )

        if "breakfast" in unit and unit["breakfast"] and unit["breakfast"] != "-":
            name, city = get_valid_name_city(unit["breakfast"])
            df = database.restaurants()
            if (
                len(
                    df[
                        (df["Name"].astype(str).str.contains(re.escape(name)))
                        & (df["City"] == city)
                    ]
                )
                < 1
            ):
                return False, f"The breakfast in day {i + 1} is invalid in the sandbox."

        if "lunch" in unit and unit["lunch"] and unit["lunch"] != "-":
            name, city = get_valid_name_city(unit["lunch"])
            df = database.restaurants()
            if (
                len(
                    df[
                        (df["Name"].astype(str).str.contains(re.escape(name)))
                        & (df["City"] == city)
                    ]
                )
                < 1
            ):
                return False, f"The lunch in day {i + 1} is invalid in the sandbox."

        if "dinner" in unit and unit["dinner"] and unit["dinner"] != "-":
            name, city = get_valid_name_city(unit["dinner"])
            df = database.restaurants()
            if (
                len(
                    df[
                        (df["Name"].astype(str).str.contains(re.escape(name)))
                        & (df["City"] == city)
                    ]
                )
                < 1
            ):
                return False, f"The dinner in day {i + 1} is invalid in the sandbox."

        if "attraction" in unit and unit["attraction"] and unit["attraction"] != "-":
            for attraction in unit["attraction"].split(";")[:-1]:
                name, city = get_valid_name_city(attraction)
                df = database.attractions()
                if (
                    len(
                        df[
                            (df["Name"].astype(str).str.contains(re.escape(name)))
                            & (df["City"] == city)
                        ]
                    )
                    < 1
                ):
                    return (
                        False,
                        f"The attraction {attraction} in day {i + 1} is invalid in the sandbox.",
                    )

        if (
            "accommodation" in unit
            and unit["accommodation"]
            and unit["accommodation"] != "-"
        ):
            name, city = get_valid_name_city(unit["accommodation"])
            df = database.accommodations()
            if (
                len(
                    df[
                        (df["NAME"].astype(str).str.contains(re.escape(name)))
                        & (df["city"] == city)
                    ]
                )
                < 1
            ):
                return (
                    False,
                    f"The accommodation in day {i + 1} is invalid in the sandbox.",
                )

    return True, None


def is_valid_accommodaton(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that each accommodation meets the minimum nights requirement.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    data = []
    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]
        if "accommodation" not in unit:
            return False, "No Accommodation Info."
        data.append(unit["accommodation"])

    consecutive_accommodation = count_consecutive_values(data)
    for unit in consecutive_accommodation:
        if unit and unit[0] not in ["-", ""]:
            name, city = get_valid_name_city(unit[0])
            df = database.accommodations()
            matches = df[
                (df["NAME"].astype(str).str.contains(re.escape(name)))
                & (df["city"] == city)
            ]
            if len(matches) == 1 and unit[1] < matches.iloc[0]["minimum nights"]:
                return (
                    False,
                    f"The accommodation {unit[0]} do not obey the minumum nights rule.",
                )

    return True, None


def is_valid_visiting_city_number(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that the correct number of destination cities are visited.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    city_set: set[str] = set()

    for i in range(min(question["days"], len(tested_data))):
        city_value = tested_data[i]["current_city"]

        if "from" in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question["org"]:
                return False, f"The first day's city should be {question['org']}."
            city_set.add(city1)
            city_set.add(city2)
        else:
            city_set.add(extract_before_parenthesis(city_value))

    city_set.discard(question["org"])

    if len(city_set) != question["visiting_city_number"]:
        return (
            False,
            f"The number of visiting cities should be {question['visiting_city_number']}.",
        )

    return True, None


def is_valid_days(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that the plan covers exactly the required number of days.

    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    lens = 0
    for i in range(min(question["days"], len(tested_data))):
        if (
            tested_data[i] != {}
            and tested_data[i]["current_city"]
            != "You don't need to fill in the information for this or later days."
        ):
            lens += 1

    if lens != question["days"]:
        return False, f"The number of days should be {question['days']}."
    return True, None


def is_not_absent(
    question: dict[str, Any], tested_data: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """Check that no critical fields are missing from the plan.

    Verifies all required keys are present and that at least 50% of fields
    are filled in. Also validates day count and visiting city count.
    Copied verbatim from evaluation/commonsense_constraint.py.
    """
    needed_info = 6 * question["days"]
    total_valid_info = 0

    if not is_valid_days(question, tested_data)[0]:
        return False, "Invalid Days"

    if not is_valid_visiting_city_number(question, tested_data)[0]:
        return False, "Invalid City Number"

    for i in range(min(question["days"], len(tested_data))):
        unit = tested_data[i]

        if "transportation" not in unit:
            return False, "No Transportation Info."
        if "breakfast" not in unit:
            return False, "No Breakfast Info."
        if "lunch" not in unit:
            return False, "No Lunch Info."
        if "dinner" not in unit:
            return False, "No Dinner Info."
        if "attraction" not in unit:
            return False, "No Attraction Info."
        if "accommodation" not in unit:
            return False, "No Accommodation Info."

        if ("from " in unit["current_city"] or "to " in unit["current_city"]) and unit[
            "transportation"
        ] in ["", "-"]:
            return False, f"No transportation in day {i + 1} is not allowed."

        if (
            "from " not in unit["current_city"] and " to " not in unit["current_city"]
        ) and unit["attraction"] in ["", "-"]:
            return False, f"No attraction in day {i + 1} is not allowed."

        if i != question["days"] - 1 and unit["accommodation"] in ["", "-"]:
            return False, f"No accommodation in day {i + 1} is not allowed."

        if (
            unit["breakfast"] in ["", "-"]
            or unit["lunch"] in ["", "-"]
            or unit["dinner"] in ["", "-"]
        ) and "from " not in unit["current_city"]:
            return False, f"No meal in day {i + 1} is not allowed."

        for key in unit:
            if unit[key] and unit[key] != "-":
                total_valid_info += 1

    # 50% completeness threshold (copied verbatim from original)
    min_completeness_ratio = 0.5
    if total_valid_info * 1.0 / needed_info < min_completeness_ratio:
        return False, "The absent information is more than 50%."

    return True, None


# ---------------------------------------------------------------------------
# Aggregate evaluation entry point
# ---------------------------------------------------------------------------


def evaluation(
    query_data: dict[str, Any], tested_data: list[dict[str, Any]]
) -> dict[str, tuple]:
    """Run all commonsense constraint checks on a parsed plan.

    Returns a dict mapping check name → (bool | None, reason | None).
    Adapted from evaluation/commonsense_constraint.py :: evaluation().
    """
    return_info: dict[str, tuple] = {}
    return_info["is_reasonable_visiting_city"] = is_reasonable_visiting_city(
        query_data, tested_data
    )
    return_info["is_valid_restaurants"] = is_valid_restaurants(query_data, tested_data)
    return_info["is_valid_attractions"] = is_valid_attractions(query_data, tested_data)
    return_info["is_valid_accommodation"] = is_valid_accommodaton(
        query_data, tested_data
    )
    return_info["is_valid_transportation"] = is_valid_transportation(
        query_data, tested_data
    )
    return_info["is_valid_information_in_current_city"] = (
        is_valid_information_in_current_city(query_data, tested_data)
    )
    return_info["is_valid_information_in_sandbox"] = is_valid_information_in_sandbox(
        query_data, tested_data
    )
    return_info["is_not_absent"] = is_not_absent(query_data, tested_data)
    return return_info
