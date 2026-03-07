"""
Lazy-loading wrappers for the TravelPlanner static database files.

Each public function returns a pandas DataFrame (or dict for city/state lookup)
loaded from the bundled CSV / text files in the database/ subdirectory.
All results are cached after the first call to avoid repeated disk I/O.
"""

import functools
from pathlib import Path

import pandas as pd
from utils import extract_before_parenthesis, extract_from_to, get_valid_name_city
import math as _math


_DB_DIR: Path = Path(__file__).parent / "database"


@functools.lru_cache(maxsize=1)
def flights() -> pd.DataFrame:
    """Return the flights DataFrame with columns used for evaluation.

    Columns: Flight Number, Price, DepTime, ArrTime, ActualElapsedTime,
             FlightDate, OriginCityName, DestCityName, Distance.
    Mirrors the loader in tools/flights/apis.py.
    """
    return pd.read_csv(_DB_DIR / "flights" / "clean_Flights_2022.csv").dropna()[
        [
            "Flight Number",
            "Price",
            "DepTime",
            "ArrTime",
            "ActualElapsedTime",
            "FlightDate",
            "OriginCityName",
            "DestCityName",
            "Distance",
        ]
    ]


@functools.lru_cache(maxsize=1)
def accommodations() -> pd.DataFrame:
    """Return the accommodations DataFrame with columns used for evaluation.

    Columns: NAME, price, room type, house_rules, minimum nights,
             maximum occupancy, review rate number, city.
    Mirrors the loader in tools/accommodations/apis.py.
    """
    return pd.read_csv(
        _DB_DIR / "accommodations" / "clean_accommodations_2022.csv"
    ).dropna()[
        [
            "NAME",
            "price",
            "room type",
            "house_rules",
            "minimum nights",
            "maximum occupancy",
            "review rate number",
            "city",
        ]
    ]


@functools.lru_cache(maxsize=1)
def restaurants() -> pd.DataFrame:
    """Return the restaurants DataFrame with columns used for evaluation.

    Columns: Name, Average Cost, Cuisines, Aggregate Rating, City.
    Mirrors the loader in tools/restaurants/apis.py.
    """
    return pd.read_csv(_DB_DIR / "restaurants" / "clean_restaurant_2022.csv").dropna()[
        ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"]
    ]


@functools.lru_cache(maxsize=1)
def attractions() -> pd.DataFrame:
    """Return the attractions DataFrame with columns used for evaluation.

    Columns: Name, Latitude, Longitude, Address, Phone, Website, City.
    Mirrors the loader in tools/attractions/apis.py.
    """
    return pd.read_csv(_DB_DIR / "attractions" / "attractions.csv").dropna()[
        ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"]
    ]


@functools.lru_cache(maxsize=1)
def distance_matrix() -> pd.DataFrame:
    """Return the full distance matrix DataFrame.

    Columns: origin, destination, duration, distance (and others).
    Mirrors the loader in tools/googleDistanceMatrix/apis.py.
    """
    return pd.read_csv(_DB_DIR / "googleDistanceMatrix" / "distance.csv")



@functools.lru_cache(maxsize=1)
def city_state_map() -> dict[str, str]:
    """Return a mapping of city name → US state name.

    Built from database/background/citySet_with_states.txt (tab-separated).
    Used in is_reasonable_visiting_city to validate destination state.
    """
    lines = (
        (_DB_DIR / "background" / "citySet_with_states.txt")
        .read_text()
        .strip()
        .split("\n")
    )
    return {
        city: state
        for city, state in (line.split("\t") for line in lines if "\t" in line)
    }


def distance_cost(org_city: str, dest_city: str, mode: str) -> int | None:
    """Look up the driving/taxi cost between two cities using the distance matrix.

    Returns None if the pair is not found, or if either city/distance/duration
    is invalid. Strips parenthetical state suffixes from city names (e.g.
    "Phoenix(AZ)" → "Phoenix") before lookup, matching the behaviour of
    googleDistanceMatrix.run_for_evaluation() from tools/googleDistanceMatrix/apis.py.
    """
    org_city = extract_before_parenthesis(org_city)
    dest_city = extract_before_parenthesis(dest_city)

    df = distance_matrix()
    response = df[(df["origin"] == org_city) & (df["destination"] == dest_city)]
    if len(response) == 0:
        return None

    row = response.iloc[0]
    dist_str = row.get("distance")
    duration = row.get("duration")
    if dist_str is None or pd.isna(dist_str):
        return None
    if duration is None or pd.isna(duration):
        return None
    if "day" in str(duration):
        return None

    km = float(str(dist_str).replace("km", "").replace(",", ""))
    if mode == "self-driving":
        return int(km * 0.05)
    elif mode == "taxi":
        return int(km)
    return None


def cost_enquiry(plan: dict) -> str:
    """Calculate the cost of a one-day sub-plan.

    Mirrors ReactEnv.run() / ReactReflectEnv.run() from tools/planner/env.py.
    Accepts the JSON dict produced by a CostEnquiry action (keys: people_number,
    transportation, breakfast, lunch, dinner, accommodation, current_city) and
    returns a human-readable cost string or an error message.

    Args:
        plan: One-day plan dict as described in REACT_PLANNER_INSTRUCTION.

    Returns:
        String of the form "The cost of your plan is N dollars." on success,
        or an error string listing reasons the cost could not be computed.
    """
    total_cost = 0.0
    errors: list[str] = []
    people = int(plan.get("people_number", 1))

    # --- Transportation ---
    transport = plan.get("transportation", "-") or "-"
    if transport != "-":
        org_city, dest_city = extract_from_to(transport)
        if org_city is None or dest_city is None:
            org_city, dest_city = extract_from_to(plan.get("current_city", ""))

        if org_city is None or dest_city is None:
            errors.append("The transportation information is not valid, please check.")
        else:
            if "flight number" in transport.lower():
                try:
                    flight_number = transport.split("Flight Number: ")[1].split(",")[0].strip()
                    df = flights()
                    res = df[df["Flight Number"] == flight_number]
                    if len(res) > 0:
                        total_cost += float(res["Price"].values[0]) * people
                    else:
                        errors.append("The flight information is not valid")
                except Exception:
                    errors.append("The flight information is not valid")
            elif "self-driving" in transport.lower() or "taxi" in transport.lower():
                mode = "self-driving" if "self-driving" in transport.lower() else "taxi"
                cost = distance_cost(org_city, dest_city, mode)
                if cost is None:
                    errors.append("The transportation information is not valid, please check.")
                elif mode == "self-driving":
                    total_cost += cost * _math.ceil(people / 5)
                else:
                    total_cost += cost * _math.ceil(people / 4)

    def _add_restaurant_cost(meal_field: str, label: str) -> None:
        value = plan.get(meal_field, "-") or "-"
        if value == "-":
            return

        name, city = get_valid_name_city(value)
        if name == "-" or city == "-":
            return
        df = restaurants()
        res = df[(df["Name"] == name) & (df["City"] == city)]
        if len(res) > 0:
            nonlocal total_cost
            total_cost += float(res["Average Cost"].values[0]) * people
        else:
            errors.append(f"The {label} information is not valid, please check.")

    _add_restaurant_cost("breakfast", "breakfast")
    _add_restaurant_cost("lunch", "lunch")
    _add_restaurant_cost("dinner", "dinner")

    # --- Accommodation ---
    accommodation = plan.get("accommodation", "-") or "-"
    if accommodation != "-":
        name, city = get_valid_name_city(accommodation)
        if name != "-" and city != "-":
            df = accommodations()
            res = df[(df["NAME"] == name) & (df["city"] == city)]
            if len(res) > 0:
                max_occ = int(res["maximum occupancy"].values[0])
                total_cost += float(res["price"].values[0]) * _math.ceil(people / max_occ)
            else:
                errors.append("The accommodation information is not valid, please check.")

    if not errors:
        return f"The cost of your plan is {total_cost} dollars."
    msg = "Sorry, the cost of your plan is not available because of the following reasons:"
    for idx, info in enumerate(errors, 1):
        msg += f"{idx}. {info} \t"
    return msg
