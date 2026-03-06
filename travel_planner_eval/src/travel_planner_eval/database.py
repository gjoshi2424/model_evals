"""
Lazy-loading wrappers for the TravelPlanner static database files.

Each public function returns a pandas DataFrame (or dict for city/state lookup)
loaded from the bundled CSV / text files in the database/ subdirectory.
All results are cached after the first call to avoid repeated disk I/O.

The bundled data comes from the original TravelPlanner repository:
  https://github.com/OSU-NLP-Group/TravelPlanner
"""

import functools
import re
from pathlib import Path

import pandas as pd

_DB_DIR: Path = Path(__file__).parent / "database"


# ---------------------------------------------------------------------------
# Lazy-loaded DataFrames
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


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


def distance_cost(org_city: str, dest_city: str, mode: str) -> float | None:
    """Look up the driving/taxi cost between two cities using the distance matrix.

    Returns None if the pair is not found, or if the duration includes 'day'.
    Replicates googleDistanceMatrix.run_for_evaluation() from tools/googleDistanceMatrix/apis.py.
    """
    df = distance_matrix()
    response = df[(df["origin"] == org_city) & (df["destination"] == dest_city)]
    if len(response) == 0:
        return None

    row = response.iloc[0]
    dist_str = row.get("distance")
    duration = row.get("duration")
    if dist_str is None or pd.isna(dist_str):
        return None
    if duration is not None and "day" in str(duration):
        return None

    km = float(re.sub(r"[^\d.]", "", str(dist_str).replace(",", "")))
    if mode == "self-driving":
        return km * 0.05
    elif mode == "taxi":
        return km * 1.0
    return None
