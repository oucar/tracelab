"""ChartSpecs are verified structured output: bad columns are rejected like bad numbers."""

from app.runtime.chartspec import extract_chart_specs

COLUMNS = ["fare", "tip", "day"]


def art(name: str, data: dict) -> dict:
    return {"kind": "json", "name": name, "data": data}


def valid_spec() -> dict:
    return {
        "kind": "bar",
        "title": "Mean fare by day",
        "x": "day",
        "y": ["mean_fare"],
        "data": [{"day": "Mon", "mean_fare": 12.1}, {"day": "Sat", "mean_fare": 15.3}],
        "source_columns": ["day", "fare"],
    }


def test_valid_spec_is_accepted():
    specs, rejections = extract_chart_specs([art("chart_fare.json", valid_spec())], COLUMNS)
    assert len(specs) == 1 and rejections == []
    assert specs[0].kind == "bar" and specs[0].y == ["mean_fare"]


def test_unknown_source_column_is_rejected():
    bad = valid_spec() | {"source_columns": ["day", "surge_multiplier"]}
    specs, rejections = extract_chart_specs([art("chart_x.json", bad)], COLUMNS)
    assert specs == [] and "surge_multiplier" in rejections[0]


def test_data_keys_must_match_axes():
    bad = valid_spec() | {"y": ["median_fare"]}
    specs, rejections = extract_chart_specs([art("chart_x.json", bad)], COLUMNS)
    assert specs == [] and "median_fare" in rejections[0]


def test_schema_violation_is_rejected_not_raised():
    specs, rejections = extract_chart_specs([art("chart_x.json", {"kind": "sunburst"})], COLUMNS)
    assert specs == [] and len(rejections) == 1


def test_non_chart_artifacts_are_ignored():
    specs, rejections = extract_chart_specs([art("result.json", {"whatever": 1})], COLUMNS)
    assert specs == [] and rejections == []


def test_empty_data_is_rejected():
    specs, rejections = extract_chart_specs(
        [art("chart_x.json", valid_spec() | {"data": []})], COLUMNS
    )
    assert specs == [] and "data" in rejections[0].lower()
