"""Tests for constraints/hard.py constraint checkers.

Database calls are patched at the module level so tests run without CSV data files.
"""

from unittest.mock import patch

import pandas as pd

from constraints.hard import (
    evaluation,
    get_total_cost,
    is_valid_cuisine,
    is_valid_room_rule,
    is_valid_room_type,
    is_valid_transportation,
)

# Shared helpers


def _question(
    org: str = "New York",
    days: int = 3,
    people_number: int = 2,
    budget: int = 5000,
    transportation: str | None = None,
    cuisine: list | None = None,
    room_type: str | None = None,
    house_rule: str | None = None,
) -> dict:
    return {
        "org": org,
        "days": days,
        "people_number": people_number,
        "budget": budget,
        "local_constraint": {
            "transportation": transportation,
            "cuisine": cuisine or [],
            "room type": room_type,
            "house rule": house_rule,
        },
    }


def _day(
    transportation: str = "-",
    current_city: str = "Los Angeles",
    breakfast: str = "-",
    lunch: str = "-",
    dinner: str = "-",
    accommodation: str = "-",
) -> dict:
    return {
        "transportation": transportation,
        "current_city": current_city,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "accommodation": accommodation,
    }


def test_valid_transportation_no_constraint():
    question = _question(transportation=None)
    data = [_day(transportation="Flight Number: AA100, from NYC to LA")]
    result, reason = is_valid_transportation(question, data)
    assert result is None
    assert reason is None


def test_valid_transportation_no_flight_passes():
    question = _question(transportation="no flight")
    data = [
        _day(transportation="Self-driving from New York to Los Angeles"),
        _day(transportation="Taxi from Los Angeles to New York"),
    ]
    result, reason = is_valid_transportation(question, data)
    assert result is True


def test_valid_transportation_no_flight_violated():
    question = _question(transportation="no flight")
    data = [_day(transportation="Flight Number: AA100, from New York to Los Angeles")]
    result, reason = is_valid_transportation(question, data)
    assert result is False
    assert "no flight" in reason


def test_valid_transportation_no_self_driving_passes():
    question = _question(transportation="no self-driving")
    data = [
        _day(transportation="Flight Number: AA100, from NYC to LA"),
        _day(transportation="Taxi from LA to NYC"),
    ]
    result, reason = is_valid_transportation(question, data)
    assert result is True


def test_valid_transportation_no_self_driving_violated():
    question = _question(transportation="no self-driving")
    data = [_day(transportation="Self-driving from New York to Los Angeles")]
    result, reason = is_valid_transportation(question, data)
    assert result is False
    assert "self-driving" in reason.lower()


def test_valid_transportation_dash_entries_ignored():
    question = _question(transportation="no flight")
    data = [_day(transportation="-"), _day(transportation="-")]
    result, reason = is_valid_transportation(question, data)
    assert result is True


def test_get_total_cost_all_dashes_is_zero():
    question = _question(people_number=2, days=2)
    data = [_day(), _day()]
    # No database calls needed — all '-' entries are skipped
    result = get_total_cost(question, data)
    assert result == 0.0


def test_get_total_cost_with_flight():
    question = _question(people_number=2, days=1)
    data = [
        _day(
            transportation="Flight Number: AA100, from New York to Los Angeles",
            current_city="from New York to Los Angeles",
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
        total = get_total_cost(question, data)
    # 300 per person × 2 people
    assert total == 600.0


def test_get_total_cost_with_self_driving():
    question = _question(people_number=4, days=1)
    data = [
        _day(
            transportation="Self-driving from New York to Los Angeles",
            current_city="from New York to Los Angeles",
        )
    ]
    # distance_cost returns 500 (mocked)
    with patch("database.distance_cost", return_value=500):
        total = get_total_cost(question, data)
    # 500 × ceil(4/5) = 500 × 1 = 500
    assert total == 500.0


def test_get_total_cost_with_taxi():
    question = _question(people_number=4, days=1)
    data = [
        _day(
            transportation="Taxi from New York to Los Angeles",
            current_city="from New York to Los Angeles",
        )
    ]
    with patch("database.distance_cost", return_value=400):
        total = get_total_cost(question, data)
    # 400 × ceil(4/4) = 400 × 1 = 400
    assert total == 400.0


def test_get_total_cost_with_restaurant():
    question = _question(people_number=2, days=1)
    data = [_day(breakfast="Nobu, Los Angeles(CA)")]
    mock_restaurants_df = pd.DataFrame(
        {
            "Name": ["Nobu"],
            "City": ["Los Angeles"],
            "Average Cost": [50],
            "Cuisines": ["Japanese"],
        }
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        total = get_total_cost(question, data)
    # 50 per person × 2 people
    assert total == 100.0


def test_get_total_cost_with_accommodation():
    question = _question(people_number=3, days=1)
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "price": [80],
            "maximum occupancy": [2],
            "minimum nights": [1],
            "house_rules": ["No pets"],
            "room type": ["Private room"],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        total = get_total_cost(question, data)
    # ceil(3 / 2) = 2 rooms × 80 = 160
    assert total == 160.0


def test_get_total_cost_unknown_restaurant_is_zero():
    question = _question(people_number=2, days=1)
    data = [_day(breakfast="Unknown Place, Nowhere(ZZ)")]
    mock_restaurants_df = pd.DataFrame(
        columns=["Name", "City", "Average Cost", "Cuisines"]
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        total = get_total_cost(question, data)
    assert total == 0.0


def test_valid_room_rule_no_constraint():
    question = _question(house_rule=None)
    data = [_day()]
    result, reason = is_valid_room_rule(question, data)
    assert result is None


def test_valid_room_rule_passes():
    question = _question(house_rule="smoking")  # "No smoking" is forbidden
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    # Hotel A has no smoking restriction
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "house_rules": ["No parties"],
            "price": [100],
            "minimum nights": [1],
            "maximum occupancy": [4],
            "room type": ["Private room"],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_rule(question, data)
    assert result is True


def test_valid_room_rule_violated():
    question = _question(house_rule="smoking")
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "house_rules": ["No smoking"],
            "price": [100],
            "minimum nights": [1],
            "maximum occupancy": [4],
            "room type": ["Private room"],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_rule(question, data)
    assert result is False
    assert "smoking" in reason


def test_valid_room_rule_all_dashes():
    question = _question(house_rule="pets")
    data = [_day(accommodation="-")]
    result, reason = is_valid_room_rule(question, data)
    assert result is True


def test_valid_cuisine_no_constraint():
    question = _question(cuisine=None)
    data = [_day()]
    result, reason = is_valid_cuisine(question, data)
    assert result is None


def test_valid_cuisine_satisfied():
    question = _question(org="New York", cuisine=["Italian"])
    data = [_day(breakfast="Trattoria, Los Angeles(CA)")]
    mock_restaurants_df = pd.DataFrame(
        {
            "Name": ["Trattoria"],
            "City": ["Los Angeles"],
            "Average Cost": [40],
            "Cuisines": ["Italian, Mediterranean"],
        }
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        result, reason = is_valid_cuisine(question, data)
    assert result is True


def test_valid_cuisine_not_satisfied():
    question = _question(org="New York", cuisine=["Italian"])
    data = [_day(breakfast="Sushi Bar, Los Angeles(CA)")]
    mock_restaurants_df = pd.DataFrame(
        {
            "Name": ["Sushi Bar"],
            "City": ["Los Angeles"],
            "Average Cost": [50],
            "Cuisines": ["Japanese"],
        }
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        result, reason = is_valid_cuisine(question, data)
    assert result is False
    assert "Italian" in reason


def test_valid_cuisine_org_city_meals_ignored():
    """Meals eaten in the origin city are excluded from cuisine checks."""
    question = _question(org="New York", cuisine=["Italian"])
    data = [_day(breakfast="Trattoria, New York(NY)")]  # in org city → ignored
    mock_restaurants_df = pd.DataFrame(
        {
            "Name": ["Trattoria"],
            "City": ["New York"],
            "Average Cost": [40],
            "Cuisines": ["Italian"],
        }
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        result, reason = is_valid_cuisine(question, data)
    assert result is False  # Italian not found outside org


def test_valid_room_type_no_constraint():
    question = _question(room_type=None)
    data = [_day()]
    result, reason = is_valid_room_type(question, data)
    assert result is None


def test_valid_room_type_private_room_passes():
    question = _question(room_type="private room")
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "room type": ["Private room"],
            "house_rules": ["No pets"],
            "price": [100],
            "minimum nights": [1],
            "maximum occupancy": [2],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_type(question, data)
    assert result is True


def test_valid_room_type_private_room_violated():
    question = _question(room_type="private room")
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "room type": ["Shared room"],
            "house_rules": ["No pets"],
            "price": [60],
            "minimum nights": [1],
            "maximum occupancy": [6],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_type(question, data)
    assert result is False
    assert "private room" in reason


def test_valid_room_type_not_shared_room_passes():
    question = _question(room_type="not shared room")
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "room type": ["Entire home/apt"],
            "house_rules": ["No pets"],
            "price": [200],
            "minimum nights": [1],
            "maximum occupancy": [6],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_type(question, data)
    assert result is True


def test_valid_room_type_not_shared_room_violated():
    question = _question(room_type="not shared room")
    data = [_day(accommodation="Hotel A, Los Angeles(CA)")]
    mock_acc_df = pd.DataFrame(
        {
            "NAME": ["Hotel A"],
            "city": ["Los Angeles"],
            "room type": ["Shared room"],
            "house_rules": ["No pets"],
            "price": [40],
            "minimum nights": [1],
            "maximum occupancy": [8],
        }
    )
    with patch("database.accommodations", return_value=mock_acc_df):
        result, reason = is_valid_room_type(question, data)
    assert result is False


def test_evaluation_returns_all_check_keys():
    question = _question()
    data = [_day()]
    result = evaluation(question, data)
    assert set(result.keys()) == {
        "valid_transportation",
        "valid_cuisine",
        "valid_room_rule",
        "valid_room_type",
        "valid_cost",
    }


def test_evaluation_all_tuples():
    question = _question()
    data = [_day()]
    result = evaluation(question, data)
    for key, value in result.items():
        assert isinstance(value, tuple), f"Expected tuple for {key}"
        assert len(value) == 2


def test_evaluation_no_constraints_all_pass():
    """With no local constraints and zero cost, everything should pass."""
    question = _question(
        budget=10000,
        transportation=None,
        cuisine=None,
        room_type=None,
        house_rule=None,
    )
    data = [_day()]  # all dashes = zero cost
    result = evaluation(question, data)
    # transportation/cuisine/room_rule/room_type → None (not applicable)
    assert result["valid_transportation"] == (None, None)
    assert result["valid_cuisine"] == (None, None)
    assert result["valid_room_rule"] == (None, None)
    assert result["valid_room_type"] == (None, None)
    # cost 0 ≤ 10000 → True
    assert result["valid_cost"][0] is True


def test_evaluation_budget_exceeded():
    """Plan costs more than budget → valid_cost should be False."""
    question = _question(budget=100, people_number=2)
    data = [_day(breakfast="Nobu, Los Angeles(CA)")]
    mock_restaurants_df = pd.DataFrame(
        {
            "Name": ["Nobu"],
            "City": ["Los Angeles"],
            "Average Cost": [200],  # 200/person × 2 = 400 > budget 100
            "Cuisines": ["Japanese"],
        }
    )
    with patch("database.restaurants", return_value=mock_restaurants_df):
        result = evaluation(question, data)
    assert result["valid_cost"][0] is False
