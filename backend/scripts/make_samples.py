"""Generate the three bundled sample datasets deterministically (seed 42).

Synthetic, modeled on real-world shapes (see data/samples/ATTRIBUTION.md).
Regenerate: python backend/scripts/make_samples.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "samples"


def _hourly_profile() -> np.ndarray:
    w = np.ones(24)
    w[[7, 8, 9, 17, 18, 19]] = 3.0
    w[[0, 1, 2, 3, 4]] = 0.4
    return w / w.sum()


def make_taxi(rng: np.random.Generator) -> pd.DataFrame:
    n = 900
    hour = rng.choice(24, size=n, p=_hourly_profile())
    day = rng.choice(DAYS, size=n)
    distance = np.round(rng.lognormal(1.1, 0.55, n), 2)
    passengers = rng.choice([1, 2, 3, 4], size=n, p=[0.62, 0.22, 0.10, 0.06])
    payment = rng.choice(["card", "cash"], size=n, p=[0.7, 0.3])
    weekend = np.isin(day, ["Sat", "Sun"])
    rush = np.isin(hour, [17, 18, 19])
    fare = 3.0 + 2.2 * distance + 2.0 * rush + 3.0 * weekend + rng.normal(0, 1.2, n)
    fare = np.round(np.clip(fare, 3.0, None), 2)
    tip = np.where(payment == "card", fare * rng.uniform(0.10, 0.25, n), rng.uniform(0, 1, n))
    return pd.DataFrame({
        "pickup_hour": hour, "day": day, "distance_km": distance,
        "passenger_count": passengers, "payment_type": payment,
        "fare": fare, "tip": np.round(tip, 2),
    })


def make_retail(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    t = np.arange(730)
    weekend = np.isin(dates.dayofweek, [5, 6])
    yearly = 60 * np.sin(2 * np.pi * (t % 365) / 365 - np.pi / 2) + 60
    promo = (rng.random(730) < np.where(weekend, 0.25, 0.10)).astype(int)
    units = 200 + 0.25 * t + 40 * weekend + yearly + 90 * promo + rng.normal(0, 25, 730)
    units = np.round(units).clip(20).astype(int)
    revenue = np.round(units * 4.5 * rng.uniform(0.95, 1.05, 730), 2)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "day": [DAYS[d] for d in dates.dayofweek],
        "units": units, "revenue": revenue, "promo": promo,
    })


def make_weather(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    t = np.arange(730)
    temp = 12 + 10 * np.sin(2 * np.pi * (t - 105) / 365.25) + rng.normal(0, 2.5, 730)
    anomaly_idx = rng.choice(730, 6, replace=False)
    temp[anomaly_idx] += rng.choice([-1.0, 1.0], 6) * 15
    humidity = np.clip(85 - 1.8 * temp + rng.normal(0, 7, 730), 20, 100)
    wind = np.round(np.abs(rng.normal(14, 6, 730)), 1)
    precip = np.round(np.where(humidity > 70, rng.exponential(4, 730),
                               rng.exponential(0.6, 730)), 1)
    condition = np.select([precip > 5, humidity > 70], ["rain", "cloudy"], default="sunny")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "temp_c": np.round(temp, 1),
        "humidity": np.round(humidity, 1), "wind_kmh": wind,
        "precip_mm": precip, "condition": condition,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    make_taxi(rng).to_csv(args.out / "taxi_trips.csv", index=False)
    make_retail(rng).to_csv(args.out / "retail_sales.csv", index=False)
    make_weather(rng).to_csv(args.out / "weather.csv", index=False)
    print(f"wrote 3 datasets to {args.out}")


if __name__ == "__main__":
    main()
