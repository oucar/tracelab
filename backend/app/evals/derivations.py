"""One derivation function per golden question — the machine-checkable ground truth.

Numeric floats are rounded to 4 decimals so YAML round-trips exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from scipy import stats

from app.evals.golden import GOLDEN_DIR, REPO_ROOT, GoldenExpected, load_golden

Derivation = Callable[[pd.DataFrame], GoldenExpected]
DERIVATIONS: dict[str, Derivation] = {}
ALPHA = 0.05
WEEKEND = ["Sat", "Sun"]


def derivation(qid: str) -> Callable[[Derivation], Derivation]:
    def register(fn: Derivation) -> Derivation:
        DERIVATIONS[qid] = fn
        return fn
    return register


def numeric(value: float, tolerance: float = 0.0) -> GoldenExpected:
    v = float(value)
    return GoldenExpected(kind="numeric", value=int(v) if v == int(v) else round(v, 4),
                         tolerance=tolerance)


def categorical(value: str) -> GoldenExpected:
    return GoldenExpected(kind="categorical", value=str(value))


def statistical(direction: str, p_value: float, method_family: str) -> GoldenExpected:
    return GoldenExpected(kind="statistical", direction=direction,
                         significant=bool(p_value < ALPHA), method_family=method_family)


# --- taxi ---------------------------------------------------------------
@derivation("taxi-001")
def _taxi_count(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("taxi-002")
def _taxi_avg_fare(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["fare"].mean(), 0.02)


@derivation("taxi-003")
def _taxi_top_hour(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int(df.groupby("pickup_hour")["fare"].mean().idxmax()))


@derivation("taxi-004")
def _taxi_busiest_day(df: pd.DataFrame) -> GoldenExpected:
    return categorical(df["day"].value_counts().idxmax())


@derivation("taxi-005")
def _taxi_max_distance(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["distance_km"].max())


@derivation("taxi-006")
def _taxi_cash_trips(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int((df["payment_type"] == "cash").sum()))


@derivation("taxi-007")
def _taxi_weekend_fare(df: pd.DataFrame) -> GoldenExpected:
    wk = df[df["day"].isin(WEEKEND)]["fare"]
    wd = df[~df["day"].isin(WEEKEND)]["fare"]
    _, p = stats.ttest_ind(wk, wd, equal_var=False)
    return statistical("higher" if wk.mean() > wd.mean() else "lower", p, "mean-comparison")


@derivation("taxi-008")
def _taxi_fare_distance_corr(df: pd.DataFrame) -> GoldenExpected:
    r, p = stats.pearsonr(df["distance_km"], df["fare"])
    return statistical("higher" if r > 0 else "lower", p, "correlation")


@derivation("taxi-009")
def _taxi_tip_by_payment(df: pd.DataFrame) -> GoldenExpected:
    card = df[df["payment_type"] == "card"]["tip"]
    cash = df[df["payment_type"] == "cash"]["tip"]
    _, p = stats.mannwhitneyu(card, cash)
    return statistical("higher" if card.mean() > cash.mean() else "lower", p, "mean-comparison")


@derivation("taxi-010")
def _taxi_median_fare(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["fare"].median(), 0.02)


# --- retail -------------------------------------------------------------
@derivation("retail-001")
def _retail_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("retail-002")
def _retail_total_revenue(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["revenue"].sum(), 0.01)


@derivation("retail-003")
def _retail_avg_units(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["units"].mean(), 0.02)


@derivation("retail-004")
def _retail_best_month(df: pd.DataFrame) -> GoldenExpected:
    months = pd.to_datetime(df["date"]).dt.month
    return numeric(int(df.groupby(months)["revenue"].mean().idxmax()))


@derivation("retail-005")
def _retail_promo_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int(df["promo"].sum()))


@derivation("retail-006")
def _retail_promo_units(df: pd.DataFrame) -> GoldenExpected:
    promo = df[df["promo"] == 1]["units"]
    rest = df[df["promo"] == 0]["units"]
    _, p = stats.ttest_ind(promo, rest, equal_var=False)
    return statistical("higher" if promo.mean() > rest.mean() else "lower", p, "mean-comparison")


@derivation("retail-007")
def _retail_promo_weekend(df: pd.DataFrame) -> GoldenExpected:
    weekend = df["day"].isin(WEEKEND)
    table = pd.crosstab(weekend, df["promo"])
    _, p, _, _ = stats.chi2_contingency(table)
    rate_wk = df[weekend]["promo"].mean()
    rate_wd = df[~weekend]["promo"].mean()
    return statistical("higher" if rate_wk > rate_wd else "lower", p, "chi-square")


@derivation("retail-008")
def _retail_trend(df: pd.DataFrame) -> GoldenExpected:
    res = stats.linregress(range(len(df)), df["units"])
    return statistical("higher" if res.slope > 0 else "lower", res.pvalue, "trend")


@derivation("retail-009")
def _retail_best_day(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["revenue"].max(), 0.01)


@derivation("retail-010")
def _retail_weekend_units(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df[df["day"].isin(WEEKEND)]["units"].mean(), 0.02)


# --- weather ------------------------------------------------------------
@derivation("weather-001")
def _weather_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("weather-002")
def _weather_max_temp(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["temp_c"].max(), 0.01)


@derivation("weather-003")
def _weather_mean_temp(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["temp_c"].mean(), 0.02)


@derivation("weather-004")
def _weather_top_condition(df: pd.DataFrame) -> GoldenExpected:
    return categorical(df["condition"].value_counts().idxmax())


@derivation("weather-005")
def _weather_temp_humidity(df: pd.DataFrame) -> GoldenExpected:
    r, p = stats.pearsonr(df["temp_c"], df["humidity"])
    return statistical("higher" if r > 0 else "lower", p, "correlation")


@derivation("weather-006")
def _weather_wind_rain(df: pd.DataFrame) -> GoldenExpected:
    rain = df[df["condition"] == "rain"]["wind_kmh"]
    sunny = df[df["condition"] == "sunny"]["wind_kmh"]
    _, p = stats.ttest_ind(rain, sunny, equal_var=False)
    return statistical("higher" if rain.mean() > sunny.mean() else "lower", p, "mean-comparison")


@derivation("weather-007")
def _weather_wet_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int((df["precip_mm"] > 10).sum()))


@derivation("weather-008")
def _weather_humidity_precip(df: pd.DataFrame) -> GoldenExpected:
    res = stats.linregress(df["humidity"], df["precip_mm"])
    return statistical("higher" if res.slope > 0 else "lower", res.pvalue, "regression")


@derivation("weather-009")
def _weather_july_temp(df: pd.DataFrame) -> GoldenExpected:
    dates = pd.to_datetime(df["date"])
    july = df[(dates.dt.year == 2024) & (dates.dt.month == 7)]
    return numeric(july["temp_c"].mean(), 0.02)


@derivation("weather-010")
def _weather_avg_wind(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["wind_kmh"].mean(), 0.02)


# --- writer -------------------------------------------------------------
def derive_all(golden_dir: Path = GOLDEN_DIR) -> dict[str, GoldenExpected]:
    out: dict[str, GoldenExpected] = {}
    for s in load_golden(golden_dir):
        df = pd.read_csv(REPO_ROOT / s.csv)
        for q in s.questions:
            if q.id in DERIVATIONS:
                derived = DERIVATIONS[q.id](df)
                derived.tolerance = q.expected.tolerance  # tolerance is authored, not derived
                out[q.id] = derived
    return out


def write_golden(golden_dir: Path = GOLDEN_DIR) -> None:
    derived = derive_all(golden_dir)
    for path in sorted(golden_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        for q in raw["questions"]:
            if q["id"] in derived:
                q["expected"] = derived[q["id"]].model_dump(exclude_defaults=False)
        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
