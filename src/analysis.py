from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from src.schema import RentalSchema
except ModuleNotFoundError:
    from .schema import RentalSchema


@dataclass(frozen=True)
class InsightMetrics:
    listing_count: int
    suburb_count: int
    median_rent: float | None
    mean_rent: float | None


def coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            cleaned = (
                result[column]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().sum() >= max(3, int(len(result) * 0.5)):
                result[column] = numeric
    return result


def compute_metrics(frame: pd.DataFrame, schema: RentalSchema) -> InsightMetrics:
    rent_series = frame[schema.rent] if schema.rent else pd.Series(dtype=float)
    suburb_series = frame[schema.suburb] if schema.suburb else pd.Series(dtype=str)
    return InsightMetrics(
        listing_count=len(frame),
        suburb_count=int(suburb_series.nunique()) if schema.suburb else 0,
        median_rent=float(rent_series.median()) if schema.rent and not rent_series.empty else None,
        mean_rent=float(rent_series.mean()) if schema.rent and not rent_series.empty else None,
    )


def build_missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().values,
            "missing_pct": (frame.isna().mean().values * 100).round(2),
            "dtype": [str(dtype) for dtype in frame.dtypes],
        }
    )
    return summary.sort_values(["missing_pct", "missing_count"], ascending=False)


def suburb_summary(frame: pd.DataFrame, schema: RentalSchema) -> pd.DataFrame:
    if not schema.suburb or not schema.rent:
        return pd.DataFrame()

    grouped = (
        frame.dropna(subset=[schema.suburb, schema.rent])
        .groupby(schema.suburb)
        .agg(
            listing_count=(schema.rent, "size"),
            median_rent=(schema.rent, "median"),
            mean_rent=(schema.rent, "mean"),
        )
        .reset_index()
        .sort_values("median_rent", ascending=False)
    )
    grouped["median_rent"] = grouped["median_rent"].round(2)
    grouped["mean_rent"] = grouped["mean_rent"].round(2)
    return grouped


def detect_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.shape[0] < 10 or numeric.shape[1] < 2:
        result = frame.copy()
        result["anomaly_flag"] = "Not enough numeric data"
        result["anomaly_score"] = np.nan
        return result

    usable = numeric.fillna(numeric.median())
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(usable)
    scores = model.decision_function(usable)

    result = frame.copy()
    result["anomaly_flag"] = np.where(predictions == -1, "Potential outlier", "Typical listing")
    result["anomaly_score"] = scores
    return result


def generate_insight_lines(frame: pd.DataFrame, schema: RentalSchema) -> list[str]:
    insights: list[str] = []

    if schema.rent and frame[schema.rent].notna().any():
        median_rent = frame[schema.rent].median()
        rent_std = frame[schema.rent].std()
        insights.append(
            f"Median weekly rent is ${median_rent:.0f}, with a standard deviation of ${rent_std:.0f}."
        )

    suburb_stats = suburb_summary(frame, schema)
    if not suburb_stats.empty:
        most_expensive = suburb_stats.iloc[0]
        cheapest = suburb_stats.iloc[-1]
        insights.append(
            f"{most_expensive[schema.suburb]} has the highest median rent in the filtered data, while {cheapest[schema.suburb]} is the lowest."
        )

    if schema.bedrooms and schema.rent:
        bedroom_rent = (
            frame.dropna(subset=[schema.bedrooms, schema.rent])
            .groupby(schema.bedrooms)[schema.rent]
            .median()
            .sort_index()
        )
        if len(bedroom_rent) >= 2:
            first = bedroom_rent.index.min()
            last = bedroom_rent.index.max()
            insights.append(
                f"Median rent rises from ${bedroom_rent.loc[first]:.0f} for {first}-bed listings to ${bedroom_rent.loc[last]:.0f} for {last}-bed listings."
            )

    numeric = frame.select_dtypes(include=[np.number])
    if schema.rent and numeric.shape[1] >= 2:
        correlations = numeric.corr(numeric_only=True)[schema.rent].drop(labels=[schema.rent], errors="ignore")
        if not correlations.empty:
            strongest = correlations.abs().sort_values(ascending=False).index[0]
            direction = "positive" if correlations[strongest] >= 0 else "negative"
            insights.append(
                f"The strongest numeric relationship with rent is a {direction} correlation with `{strongest}` ({correlations[strongest]:.2f})."
            )

    if not insights:
        insights.append("Upload a dataset with rental-related numeric and categorical columns to generate insights.")

    return insights
