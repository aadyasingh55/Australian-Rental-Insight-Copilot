from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    build_missingness_table,
    coerce_numeric_columns,
    compute_metrics,
    detect_outliers,
    generate_insight_lines,
    suburb_summary,
)
from src.schema import RentalSchema, infer_schema, normalize_columns


PROJECT_ROOT = Path(__file__).parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_rental_listings.csv"


@st.cache_data
def load_frame(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        frame = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        frame = pd.read_csv(uploaded_file)
    frame = normalize_columns(frame)
    return coerce_numeric_columns(frame)


def apply_filters(frame: pd.DataFrame, schema: RentalSchema) -> pd.DataFrame:
    filtered = frame.copy()

    st.sidebar.header("Filters")
    if schema.suburb and filtered[schema.suburb].notna().any():
        suburbs = sorted(filtered[schema.suburb].dropna().astype(str).unique().tolist())
        selected_suburbs = st.sidebar.multiselect("Suburb", suburbs, default=suburbs)
        if selected_suburbs:
            filtered = filtered[filtered[schema.suburb].astype(str).isin(selected_suburbs)]

    if schema.property_type and filtered[schema.property_type].notna().any():
        property_types = sorted(filtered[schema.property_type].dropna().astype(str).unique().tolist())
        selected_types = st.sidebar.multiselect("Property type", property_types, default=property_types)
        if selected_types:
            filtered = filtered[filtered[schema.property_type].astype(str).isin(selected_types)]

    if schema.bedrooms and pd.api.types.is_numeric_dtype(filtered[schema.bedrooms]):
        bedrooms = sorted(filtered[schema.bedrooms].dropna().unique().tolist())
        selected_bedrooms = st.sidebar.multiselect("Bedrooms", bedrooms, default=bedrooms)
        if selected_bedrooms:
            filtered = filtered[filtered[schema.bedrooms].isin(selected_bedrooms)]

    if schema.rent and pd.api.types.is_numeric_dtype(filtered[schema.rent]):
        min_rent = float(filtered[schema.rent].min())
        max_rent = float(filtered[schema.rent].max())
        rent_range = st.sidebar.slider(
            "Weekly rent range",
            min_value=float(min_rent),
            max_value=float(max_rent),
            value=(float(min_rent), float(max_rent)),
        )
        filtered = filtered[filtered[schema.rent].between(rent_range[0], rent_range[1])]

    return filtered


def render_metrics(frame: pd.DataFrame, schema: RentalSchema) -> None:
    metrics = compute_metrics(frame, schema)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Listings", metrics.listing_count)
    col2.metric("Suburbs", metrics.suburb_count)
    col3.metric("Median rent", f"${metrics.median_rent:.0f}" if metrics.median_rent else "N/A")
    col4.metric("Average rent", f"${metrics.mean_rent:.0f}" if metrics.mean_rent else "N/A")


def render_overview(frame: pd.DataFrame, schema: RentalSchema) -> None:
    st.subheader("Insight summary")
    for line in generate_insight_lines(frame, schema):
        st.write(f"- {line}")

    st.subheader("Statistical snapshot")
    st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Missing values")
    st.dataframe(build_missingness_table(frame), use_container_width=True, hide_index=True)


def render_visuals(frame: pd.DataFrame, schema: RentalSchema) -> None:
    if schema.rent and pd.api.types.is_numeric_dtype(frame[schema.rent]):
        st.subheader("Weekly rent distribution")
        st.plotly_chart(
            px.histogram(frame, x=schema.rent, nbins=25, title="Weekly rent distribution"),
            use_container_width=True,
        )

    if schema.suburb and schema.rent:
        suburb_table = suburb_summary(frame, schema).head(10)
        if not suburb_table.empty:
            st.subheader("Top suburbs by median rent")
            st.plotly_chart(
                px.bar(
                    suburb_table.sort_values("median_rent"),
                    x="median_rent",
                    y=schema.suburb,
                    orientation="h",
                    title="Top 10 suburbs by median weekly rent",
                ),
                use_container_width=True,
            )

    if schema.property_type and schema.rent:
        st.subheader("Rent spread by property type")
        st.plotly_chart(
            px.box(frame, x=schema.property_type, y=schema.rent, color=schema.property_type),
            use_container_width=True,
        )

    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) >= 2:
        st.subheader("Custom multivariable scatter plot")
        col1, col2, col3 = st.columns(3)
        x_axis = col1.selectbox("X axis", numeric_columns, index=0)
        y_axis = col2.selectbox("Y axis", numeric_columns, index=min(1, len(numeric_columns) - 1))
        color_options = ["None"] + frame.columns.tolist()
        color_choice = col3.selectbox("Colour by", color_options, index=0)

        scatter_args = {"x": x_axis, "y": y_axis}
        if color_choice != "None":
            scatter_args["color"] = color_choice

        st.plotly_chart(
            px.scatter(frame, **scatter_args, title=f"{y_axis} vs {x_axis}"),
            use_container_width=True,
        )


def render_correlations(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        st.info("Add at least two numeric columns to view correlations.")
        return

    corr = numeric.corr(numeric_only=True)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        )
    )
    heatmap.update_layout(title="Numeric feature correlation matrix")
    st.plotly_chart(heatmap, use_container_width=True)
    st.dataframe(corr.round(3), use_container_width=True)


def render_outliers(frame: pd.DataFrame, schema: RentalSchema) -> None:
    with_flags = detect_outliers(frame)
    flagged = with_flags[with_flags["anomaly_flag"] == "Potential outlier"].copy()
    st.subheader("Anomaly detection")
    st.write(
        "Listings flagged here are statistically unusual relative to the numeric profile of the filtered dataset."
    )
    st.dataframe(with_flags[["anomaly_flag", "anomaly_score"] + frame.columns.tolist()], use_container_width=True)

    if schema.rent and schema.suburb and not flagged.empty:
        st.plotly_chart(
            px.scatter(
                with_flags,
                x=schema.suburb,
                y=schema.rent,
                color="anomaly_flag",
                title="Potential outliers by suburb and weekly rent",
            ),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Australian Rental Insight Copilot",
        page_icon="🏘️",
        layout="wide",
    )
    st.title("Australian Rental Insight Copilot")
    st.caption(
        "Upload a rental dataset to explore pricing patterns, suburb comparisons, correlations, and unusual listings."
    )

    uploaded_file = st.file_uploader("Upload rental CSV", type=["csv"])
    frame = load_frame(uploaded_file)
    schema = infer_schema(frame)
    filtered = apply_filters(frame, schema)

    st.caption(
        "Sample dataset is loaded by default. Upload your own CSV to replace it."
    )
    if not schema.rent or not schema.suburb:
        st.warning(
            "This app works best with rental-style CSVs that include suburb/location and weekly rent columns. "
            "If your dataset uses very different field names or mixed text values, some charts will be limited."
        )

    if filtered.empty:
        st.warning("The current filters return no rows. Adjust the sidebar filters to continue.")
        with st.expander("Inferred schema"):
            st.json(schema.__dict__)
        return

    render_metrics(filtered, schema)

    with st.expander("Inferred schema"):
        st.json(schema.__dict__)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visuals", "Correlations", "Outliers"])
    with tab1:
        render_overview(filtered, schema)
    with tab2:
        render_visuals(filtered, schema)
    with tab3:
        render_correlations(filtered)
    with tab4:
        render_outliers(filtered, schema)


if __name__ == "__main__":
    main()
