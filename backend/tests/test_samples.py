import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "data" / "samples"

EXPECTED_COLUMNS = {
    "taxi_trips.csv": ["pickup_hour", "day", "distance_km", "passenger_count",
                       "payment_type", "fare", "tip"],
    "retail_sales.csv": ["date", "day", "units", "revenue", "promo"],
    "weather.csv": ["date", "temp_c", "humidity", "wind_kmh", "precip_mm", "condition"],
}


def test_generator_is_deterministic(tmp_path):
    script = REPO / "backend" / "scripts" / "make_samples.py"
    for out in (tmp_path / "a", tmp_path / "b"):
        subprocess.run([sys.executable, str(script), "--out", str(out)], check=True)
    for name in EXPECTED_COLUMNS:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_committed_samples_match_schema():
    for name, cols in EXPECTED_COLUMNS.items():
        df = pd.read_csv(SAMPLES / name)
        assert list(df.columns) == cols
        assert len(df) in (900, 730)


def test_taxi_has_designed_effects():
    df = pd.read_csv(SAMPLES / "taxi_trips.csv")
    weekend = df[df["day"].isin(["Sat", "Sun"])]["fare"].mean()
    weekday = df[~df["day"].isin(["Sat", "Sun"])]["fare"].mean()
    assert weekend > weekday
    assert df["fare"].corr(df["distance_km"]) > 0.6
