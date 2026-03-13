"""Tests for dataset.py — record parsing and Sample construction."""

import pytest

from dataset import record_to_sample
from prompts import COT_PLANNER_INSTRUCTION, PLANNER_INSTRUCTION


_BASE_RECORD: dict = {
    "org": "New York",
    "dest": "California",
    "days": 3,
    "visiting_city_number": 1,
    "people_number": 2,
    "budget": 5000,
    "local_constraint": {
        "transportation": None,
        "cuisine": [],
        "room type": None,
        "house rule": None,
    },
    "level": "easy",
    "query": "Plan a 3-day trip from New York to California for 2 people.",
    "reference_information": "Flight AA100 departs JFK at 08:00, arrives LAX at 11:30.",
    "date": "2024-03-15",
}


@pytest.fixture
def base_record() -> dict:
    return _BASE_RECORD.copy()


def test_sample_input_contains_query(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert base_record["query"] in sample.input


def test_sample_input_contains_reference_information(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert base_record["reference_information"] in sample.input


def test_sample_input_uses_cot_instruction():
    sample = record_to_sample(_BASE_RECORD, COT_PLANNER_INSTRUCTION)
    # COT instruction template includes the reference text and query too
    assert _BASE_RECORD["query"] in sample.input
    assert _BASE_RECORD["reference_information"] in sample.input

def test_sample_metadata_org(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["org"] == "New York"


def test_sample_metadata_dest(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["dest"] == "California"


def test_sample_metadata_days(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["days"] == 3


def test_sample_metadata_visiting_city_number(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["visiting_city_number"] == 1


def test_sample_metadata_people_number(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["people_number"] == 2


def test_sample_metadata_budget(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["budget"] == 5000


def test_sample_metadata_level(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert sample.metadata["level"] == "easy"


def test_sample_metadata_local_constraint(base_record):
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    lc = sample.metadata["local_constraint"]
    assert lc["transportation"] is None
    assert lc["cuisine"] == []
    assert lc["room type"] is None
    assert lc["house rule"] is None


def test_sample_metadata_contains_raw_fields(base_record):
    """reference_information and query must be stored for react/reflexion solvers."""
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    assert "reference_information" in sample.metadata
    assert "query" in sample.metadata

def test_record_local_constraint_as_string_is_parsed(base_record):
    base_record["local_constraint"] = (
        "{'transportation': 'no flight', 'cuisine': ['Italian'], "
        "'room type': 'private room', 'house rule': None}"
    )
    sample = record_to_sample(base_record, PLANNER_INSTRUCTION)
    lc = sample.metadata["local_constraint"]
    assert lc["transportation"] == "no flight"
    assert lc["cuisine"] == ["Italian"]
    assert lc["room type"] == "private room"
    assert lc["house rule"] is None
