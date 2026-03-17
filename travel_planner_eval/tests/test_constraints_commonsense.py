"""Tests for constraints/commonsense.py constraint checkers.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd

from constraints.commonsense import (
    evaluation,
    is_not_absent,
    is_reasonable_visiting_city,
    is_valid_accommodation,
    is_valid_attractions,
    is_valid_days,
    is_valid_information_in_current_city,
    is_valid_information_in_sandbox,
    is_valid_restaurants,
    is_valid_transportation,
    is_valid_visiting_city_number,
)

# A minimal city→state mapping used for is_reasonable_visiting_city tests.
CITY_STATE = {
    "New York": "New York",
    "Los Angeles": "California",
    "San Francisco": "California",
    "Chicago": "Illinois",
}


@contextmanager
def patch_all_databases(city_state=None):
    """Patch every database call with empty DataFrames (and optionally city_state_map).

    Use this in any test that calls evaluation() or is_valid_information_in_sandbox()
    without real CSV data, so that all database paths are isolated.
    """
    empty_restaurants = pd.DataFrame(
        columns=["Name", "City", "Average Cost", "Cuisines"]
    )
    empty_flights = pd.DataFrame(
        columns=["Flight Number", "OriginCityName", "DestCityName", "Price"]
    )
    empty_attractions = pd.DataFrame(columns=["Name", "City"])
    empty_accommodations = pd.DataFrame(
        columns=[
            "NAME",
            "city",
            "minimum nights",
            "price",
            "maximum occupancy",
            "house_rules",
            "room type",
        ]
    )
    with (
        patch("database.city_state_map", return_value=city_state or CITY_STATE),
        patch("database.flights", return_value=empty_flights),
        patch("database.restaurants", return_value=empty_restaurants),
        patch("database.attractions", return_value=empty_attractions),
        patch("database.accommodations", return_value=empty_accommodations),
    ):
        yield


def day_dict(
    current_city: str = "Los Angeles",
    transportation: str = "-",
    breakfast: str = "-",
    lunch: str = "-",
    dinner: str = "-",
    attraction: str = "-",
    accommodation: str = "-",
) -> dict:
    """Return a minimal per-day_dict plan dict with sensible defaults."""
    return {
        "current_city": current_city,
        "transportation": transportation,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "attraction": attraction,
        "accommodation": accommodation,
    }


def question_dict(
    org: str = "New York",
    dest: str = "California",
    days: int = 3,
    visiting_city_number: int = 1,
    people_number: int = 2,
    budget: int = 5000,
) -> dict:
    return {
        "org": org,
        "dest": dest,
        "days": days,
        "visiting_city_number": visiting_city_number,
        "people_number": people_number,
        "budget": budget,
    }


def test_reasonable_visiting_city_valid_round_trip():
    question = question_dict(org="New York", dest="California", days=3)
    data = [
        day_dict("from New York to Los Angeles"),
        day_dict("Los Angeles"),
        day_dict("from Los Angeles to New York"),
    ]
    with patch("database.city_state_map", return_value=CITY_STATE):
        result, reason = is_reasonable_visiting_city(question, data)
    assert result is True
    assert reason is None


def test_reasonable_visiting_city_wrong_first_city():
    question = question_dict(org="New York", dest="California", days=1)
    data = [day_dict("from Chicago to Los Angeles")]
    with patch("database.city_state_map", return_value=CITY_STATE):
        result, reason = is_reasonable_visiting_city(question, data)
    assert result is False
    assert "New York" in reason


def test_reasonable_visiting_city_not_closed_circle():
    question = question_dict(org="New York", dest="California", days=2)
    data = [
        day_dict("from New York to Los Angeles"),
        day_dict("Los Angeles"),
    ]
    with patch("database.city_state_map", return_value=CITY_STATE):
        result, reason = is_reasonable_visiting_city(question, data)
    assert result is False
    assert "closed circle" in reason


def test_reasonable_visiting_city_unknown_city():
    question = question_dict(org="New York", dest="California", days=2)
    data = [
        day_dict("from New York to Atlantis"),
        day_dict("from Atlantis to New York"),
    ]
    # Atlantis is not in city_state_map
    with patch("database.city_state_map", return_value=CITY_STATE):
        result, reason = is_reasonable_visiting_city(question, data)
    assert result is False
    assert "Atlantis" in reason


def test_valid_restaurants_no_repeats():
    question = question_dict(days=2)
    data = [
        day_dict(breakfast="Cafe A, Los Angeles(CA)", lunch="Diner B, Los Angeles(CA)"),
        day_dict(breakfast="Cafe C, Los Angeles(CA)", dinner="Diner D, Los Angeles(CA)"),
    ]
    result, reason = is_valid_restaurants(question, data)
    assert result is True


def test_valid_restaurants_repeated_breakfast():
    question = question_dict(days=2)
    data = [
        day_dict(breakfast="Cafe A, Los Angeles(CA)"),
        day_dict(breakfast="Cafe A, Los Angeles(CA)"),
    ]
    result, reason = is_valid_restaurants(question, data)
    assert result is False
    assert "repeated" in reason


def test_valid_restaurants_repeated_lunch():
    question = question_dict(days=2)
    data = [
        day_dict(lunch="Diner B, Los Angeles(CA)"),
        day_dict(lunch="Diner B, Los Angeles(CA)"),
    ]
    result, reason = is_valid_restaurants(question, data)
    assert result is False


def test_valid_restaurants_all_dashes():
    question = question_dict(days=2)
    data = [day_dict(), day_dict()]
    result, reason = is_valid_restaurants(question, data)
    assert result is True


def test_valid_attractions_no_repeats():
    question = question_dict(days=2)
    data = [
        day_dict(attraction="Griffith Observatory;"),
        day_dict(attraction="Getty Center;"),
    ]
    result, reason = is_valid_attractions(question, data)
    assert result is True


def test_valid_attractions_repeated():
    question = question_dict(days=2)
    data = [
        day_dict(attraction="Griffith Observatory;"),
        day_dict(attraction="Griffith Observatory;"),
    ]
    result, reason = is_valid_attractions(question, data)
    assert result is False
    assert "repeated" in reason


def test_valid_attractions_all_dashes():
    question = question_dict(days=2)
    data = [day_dict(), day_dict()]
    result, reason = is_valid_attractions(question, data)
    assert result is True


def test_valid_transportation_self_driving_only():
    question = question_dict(days=2)
    data = [
        day_dict(transportation="Self-driving from New York to Los Angeles"),
        day_dict(transportation="Self-driving from Los Angeles to New York"),
    ]
    result, reason = is_valid_transportation(question, data)
    assert result is True


def test_valid_transportation_flight_and_self_driving_conflict():
    question = question_dict(days=2)
    data = [
        day_dict(transportation="Flight Number: AA100, from New York to Los Angeles"),
        day_dict(transportation="Self-driving from Los Angeles to New York"),
    ]
    result, reason = is_valid_transportation(question, data)
    assert result is False
    assert "conflicting" in reason


def test_valid_transportation_taxi_and_self_driving_conflict():
    question = question_dict(days=2)
    data = [
        day_dict(transportation="Taxi from New York to Los Angeles"),
        day_dict(transportation="Self-driving from Los Angeles to New York"),
    ]
    result, reason = is_valid_transportation(question, data)
    assert result is False


def test_valid_transportation_empty_day1():
    question = question_dict(days=2)
    data = [day_dict(transportation="-"), day_dict(transportation="Taxi from A to B")]
    result, reason = is_valid_transportation(question, data)
    assert result is False
    assert "day 1" in reason


def test_valid_info_in_current_city_valid():
    question = question_dict(days=1)
    data = [
        day_dict(
            current_city="from New York to Los Angeles",
            transportation="Self-driving from New York to Los Angeles",
            breakfast="Nobu, Los Angeles(CA)",
            accommodation="Hotel A, Los Angeles(CA)",
        )
    ]
    result, reason = is_valid_information_in_current_city(question, data)
    assert result is True


def test_valid_info_in_current_city_wrong_city_breakfast():
    question = question_dict(days=1)
    data = [
        day_dict(
            current_city="Los Angeles",
            breakfast="Nobu, Chicago(IL)",  # wrong city
        )
    ]
    result, reason = is_valid_information_in_current_city(question, data)
    assert result is False
    assert "breakfast" in reason


def test_valid_info_in_current_city_wrong_accommodation():
    question = question_dict(days=1)
    data = [
        day_dict(
            current_city="Los Angeles",
            accommodation="Marriott, Chicago(IL)",  # wrong city
        )
    ]
    result, reason = is_valid_information_in_current_city(question, data)
    assert result is False
    assert "accommodation" in reason


def test_valid_info_in_sandbox_all_dashes():
    """Plan with all '-' entries should pass without any database lookups."""
    question = question_dict(days=2)
    data = [day_dict(), day_dict()]
    result, reason = is_valid_information_in_sandbox(question, data)
    assert result is True


def test_valid_info_in_sandbox_valid_flight():
    question = question_dict(days=1)
    data = [
        day_dict(
            current_city="from New York to Los Angeles",
            transportation="Flight Number: AA100, from New York to Los Angeles",
        )
    ]
    mock_flights_df = pd.DataFrame(
        {
            "Flight Number": ["AA100"],
            "OriginCityName": ["New York"],
            "DestCityName": ["Los Angeles"],
            "Price": [300],
        }
    )
    with patch("database.flights", return_value=mock_flights_df):
        result, reason = is_valid_information_in_sandbox(question, data)
    assert result is True


def test_valid_info_in_sandbox_invalid_flight():
    question = question_dict(days=1)
    data = [
        day_dict(
            current_city="from New York to Los Angeles",
            transportation="Flight Number: ZZ999, from New York to Los Angeles",
        )
    ]
    # Empty DataFrame — flight not found
    mock_flights_df = pd.DataFrame(
        columns=["Flight Number", "OriginCityName", "DestCityName", "Price"]
    )
    with patch("database.flights", return_value=mock_flights_df):
        result, reason = is_valid_information_in_sandbox(question, data)
    assert result is False
    assert "flight" in reason.lower()


def test_valid_info_in_sandbox_invalid_restaurant():
    question = question_dict(days=1)
    data = [day_dict(breakfast="Unknown Place, Los Angeles(CA)")]
    mock_restaurants_df = pd.DataFrame(
        columns=["Name", "City", "Average Cost", "Cuisines"]
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        result, reason = is_valid_information_in_sandbox(question, data)
    assert result is False
    assert "breakfast" in reason


def test_valid_accommodaton_meets_minimum_nights():
    question = question_dict(days=2)
    data = [
        day_dict(accommodation="Hotel A, Los Angeles(CA)"),
        day_dict(accommodation="Hotel A, Los Angeles(CA)"),
    ]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "minimum nights": [2],
            "price": [100],
            "maximum occupancy": [4],
            "house_rules": ["No pets"],
            "room type": ["Private room"],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_accommodation(question, data)
    assert result is True


def test_valid_accommodaton_violates_minimum_nights():
    # Only 1 night but minimum is 2
    question = question_dict(days=2)
    data = [
        day_dict(accommodation="Hotel A, Los Angeles(CA)"),
        day_dict(accommodation="Hotel B, Los Angeles(CA)"),  # different hotel
    ]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "minimum nights": [2],
            "price": [100],
            "maximum occupancy": [4],
            "house_rules": ["No pets"],
            "room type": ["Private room"],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_accommodation(question, data)
    assert result is False
    assert "minimum" in reason.lower() or "nights" in reason.lower()


def test_valid_visiting_city_number_correct():
    question = question_dict(org="New York", days=3, visiting_city_number=1)
    data = [
        day_dict("from New York to Los Angeles"),
        day_dict("Los Angeles"),
        day_dict("from Los Angeles to New York"),
    ]
    result, reason = is_valid_visiting_city_number(question, data)
    assert result is True


def test_valid_visiting_city_number_wrong_count():
    # Plan visits 2 cities but constraint says 1
    question = question_dict(org="New York", days=4, visiting_city_number=1)
    data = [
        day_dict("from New York to Los Angeles"),
        day_dict("Los Angeles"),
        day_dict("from Los Angeles to San Francisco"),
        day_dict("from San Francisco to New York"),
    ]
    result, reason = is_valid_visiting_city_number(question, data)
    assert result is False
    assert "visiting" in reason.lower() or "cities" in reason.lower()


def test_valid_visiting_city_number_wrong_first_city():
    question = question_dict(org="New York", days=1, visiting_city_number=1)
    data = [day_dict("from Chicago to Los Angeles")]
    result, reason = is_valid_visiting_city_number(question, data)
    assert result is False
    assert "New York" in reason


def test_valid_days_correct():
    question = question_dict(days=2)
    data = [day_dict("Los Angeles"), day_dict("Los Angeles")]
    result, reason = is_valid_days(question, data)
    assert result is True


def test_valid_days_too_few():
    question = question_dict(days=3)
    data = [day_dict("Los Angeles"), day_dict("Los Angeles")]
    result, reason = is_valid_days(question, data)
    assert result is False
    assert "3" in reason


def test_valid_days_placeholder_row():
    question = question_dict(days=2)
    data = [
        day_dict("Los Angeles"),
        day_dict("You don't need to fill in the information for this or later days."),
    ]
    result, reason = is_valid_days(question, data)
    assert result is False


def test_not_absent_valid_plan():
    question = question_dict(org="New York", days=3, visiting_city_number=1)
    data = [
        {
            "current_city": "from New York to Los Angeles",
            "transportation": "Self-driving from New York to Los Angeles",
            "breakfast": "-",
            "lunch": "-",
            "dinner": "-",
            "attraction": "-",
            "accommodation": "Hotel A, Los Angeles",
        },
        {
            "current_city": "Los Angeles",
            "transportation": "-",
            "breakfast": "Cafe A, Los Angeles",
            "lunch": "Diner B, Los Angeles",
            "dinner": "Restaurant C, Los Angeles",
            "attraction": "Getty Center;",
            "accommodation": "Hotel A, Los Angeles",
        },
        {
            "current_city": "from Los Angeles to New York",
            "transportation": "Self-driving from Los Angeles to New York",
            "breakfast": "-",
            "lunch": "-",
            "dinner": "-",
            "attraction": "-",
            "accommodation": "-",
        },
    ]
    result, reason = is_not_absent(question, data)
    assert result is True, f"Expected True but got False: {reason}"


def test_not_absent_missing_transportation_key():
    question = question_dict(days=1)
    data = [
        {
            "current_city": "Los Angeles",
            # "transportation" key intentionally missing
            "breakfast": "Cafe, LA",
            "lunch": "Diner, LA",
            "dinner": "Rest, LA",
            "attraction": "Park;",
            "accommodation": "Hotel, LA",
        }
    ]
    result, reason = is_not_absent(question, data)
    assert result is False
    assert "Transportation" in reason


def test_evaluation_returns_all_check_keys():
    question = question_dict(
        org="New York", dest="California", days=3, visiting_city_number=1
    )
    data = [
        day_dict("from New York to Los Angeles"),
        day_dict("Los Angeles"),
        day_dict("from Los Angeles to New York"),
    ]
    with patch_all_databases():
        results = evaluation(question, data)

    expected_keys = {
        "is_reasonable_visiting_city",
        "is_valid_restaurants",
        "is_valid_attractions",
        "is_valid_transportation",
        "is_valid_information_in_current_city",
        "is_valid_information_in_sandbox",
        "is_valid_accommodation",
        "is_not_absent",
    }
    assert expected_keys == set(results.keys())


def test_evaluation_returns_tuples():
    question = question_dict(
        org="New York", dest="California", days=1, visiting_city_number=1
    )
    data = [day_dict("from New York to Los Angeles")]
    with patch_all_databases():
        results = evaluation(question, data)

    for key, value in results.items():
        assert isinstance(value, tuple), f"Expected tuple for {key}, got {type(value)}"
        assert len(value) == 2, f"Expected 2-element tuple for {key}"
