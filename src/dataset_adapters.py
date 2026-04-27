from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import normalize_columns


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_columns(frame)


def load_australian_rental_market_data(path: str | Path) -> pd.DataFrame:
    frame = load_csv(path)
    rename_map = {
        "price": "weekly_rent",
        "weekly_rent_aud": "weekly_rent",
        "property_sub_type": "property_type",
    }
    existing = {key: value for key, value in rename_map.items() if key in frame.columns}
    if existing:
        frame = frame.rename(columns=existing)
    return frame


def load_nsw_bond_lodgement_data(path: str | Path) -> pd.DataFrame:
    frame = load_csv(path)
    rename_map = {
        "weekly_rent": "weekly_rent",
        "dwelling_type": "property_type",
        "postcode": "postcode",
        "bedrooms": "bedrooms",
        "lodgement_date": "lodgement_date",
        "month": "lodgement_month",
    }
    existing = {key: value for key, value in rename_map.items() if key in frame.columns}
    if existing:
        frame = frame.rename(columns=existing)
    return frame
