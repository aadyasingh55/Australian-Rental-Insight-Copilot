# Australian-Rental-Insight-Copilot

A Streamlit app for exploring Australian rental market datasets with automated statistical analysis, rental-focused visualisations, multivariable plotting, and anomaly detection.

## Overview

This project is designed around real Australian housing data rather than a generic CSV dashboard. It focuses on practical rental-market questions such as:

- Which suburbs are more affordable for renters?
- How do bedrooms, property type, and location affect weekly rent?
- Which listings or postcode groups look unusually expensive relative to the rest of the market?

## Real dataset strategy

The app is structured to work with two stronger public datasets:

**Primary dataset**
- [Australian Rental Market Data 2026](https://www.kaggle.com/datasets/kanchana1990/australian-rental-market-data-2026)

**Secondary dataset**
- [NSW Rental Bond Lodgement data](https://www.nsw.gov.au/housing-and-construction/rental-forms-surveys-and-data/rental-bond-data)

This gives the project:

- a national cleaned rental listing dataset for broad EDA
- an official NSW government dataset for validation and trend-style analysis

## Features

- CSV upload with automatic column-type detection
- rental-focused field inference for rent, suburb, bedrooms, bathrooms, property type, and postcode
- summary statistics and missing-value analysis
- filtering by suburb, property type, bedrooms, and rent range
- histograms, box plots, suburb comparisons, and multivariable scatter plots
- correlation matrix for numeric features
- anomaly detection using `IsolationForest`
- plain-English insight summary generated from dataset statistics

## Tech stack

- Python
- Streamlit
- pandas
- Plotly
- scikit-learn

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── external/
│   └── sample_rental_listings.csv
└── src/
    ├── __init__.py
    ├── analysis.py
    ├── dataset_adapters.py
    └── schema.py
```

## How to run

```bash
git clone https://github.com/aadyasingh55/Australian-Rental-Insight-Copilot.git
cd Australian-Rental-Insight-Copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## External dataset setup

Place one or both real datasets in `data/external/` using these filenames:

```text
data/external/australian_rental_market_2026.csv
data/external/nsw_rental_bond_lodgement.csv
```

The app will automatically:

- prefer the Australian Rental Market dataset if present
- otherwise fall back to the NSW bond lodgement dataset
- otherwise use the bundled sample CSV as a backup

## Dataset expectations

The app works best with rental-style CSV files containing columns similar to:

- `weekly_rent`
- `suburb`
- `bedrooms`
- `bathrooms`
- `property_type`
- `postcode`

The bundled sample CSV exists only as a fallback so the app can still run without external data.

## Current scope

This repository is an MVP analytics tool rather than a full production rental intelligence platform.

- It works best on relatively clean tabular rental listing data
- Schema inference is heuristic, so some real-world exports may need minor cleanup
- The anomaly detection and insight summaries are exploratory signals, not final market judgments

## Resume-ready description

Built an Australian rental market insight copilot that performs automated EDA, multivariable visualisation, anomaly detection, and rental pricing analysis using a national rental listing dataset and official NSW bond lodgement data in a Streamlit interface.
