from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class RentalSchema:
    rent: str | None
    suburb: str | None
    bedrooms: str | None
    bathrooms: str | None
    property_type: str | None
    postcode: str | None


RENT_PATTERNS = ("rent", "price", "weekly", "pw", "per_week")
SUBURB_PATTERNS = ("suburb", "area", "location")
BED_PATTERNS = ("bed", "bedroom", "beds")
BATH_PATTERNS = ("bath", "bathroom", "baths")
TYPE_PATTERNS = ("property_type", "type", "dwelling")
POSTCODE_PATTERNS = ("postcode", "post_code", "zip")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in renamed.columns
    ]
    return renamed


def _match_column(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    for column in columns:
        if any(pattern in column for pattern in patterns):
            return column
    return None


def infer_schema(frame: pd.DataFrame) -> RentalSchema:
    columns = list(frame.columns)
    return RentalSchema(
        rent=_match_column(columns, RENT_PATTERNS),
        suburb=_match_column(columns, SUBURB_PATTERNS),
        bedrooms=_match_column(columns, BED_PATTERNS),
        bathrooms=_match_column(columns, BATH_PATTERNS),
        property_type=_match_column(columns, TYPE_PATTERNS),
        postcode=_match_column(columns, POSTCODE_PATTERNS),
    )
