
from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "raw" / "crop_yield.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# EXPECTED DATASET COLUMNS
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Please place crop_yield.csv inside dataset/raw/"
        )

    df = pd.read_csv(DATASET_PATH)

    # Remove accidental spaces in column names
    df.columns = df.columns.str.strip()

    return df


# --------------------------------------------------
# DATASET DIMENSIONS
# --------------------------------------------------

def inspect_shape(df):

    print("\n========== DATASET SHAPE ==========")

    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])


# --------------------------------------------------
# COLUMN INFORMATION
# --------------------------------------------------

def inspect_columns(df):

    print("\n========== COLUMN NAMES ==========")

    for column in df.columns:
        print("-", column)

    print("\n========== DATA TYPES ==========")

    print(df.dtypes)


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

def inspect_missing_values(df):

    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()

    print(missing)

    missing.to_csv(
        OUTPUT_DIR / "missing_values.csv"
    )


# --------------------------------------------------
# DUPLICATE RECORDS
# --------------------------------------------------

def inspect_duplicates(df):

    print("\n========== DUPLICATES ==========")

    duplicates = df.duplicated().sum()

    print("Duplicate records:", duplicates)


# --------------------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------------------

def inspect_statistics(df):

    print("\n========== NUMERICAL SUMMARY ==========")

    print(
        df.describe(include=[np.number]).T
    )

    df.describe(
        include="all"
    ).T.to_csv(
        OUTPUT_DIR / "dataset_summary.csv"
    )


# --------------------------------------------------
# CATEGORICAL INFORMATION
# --------------------------------------------------

def inspect_categories(df):

    print("\n========== CATEGORICAL FEATURES ==========")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        print(f"\n{column}")

        print("Unique values:", df[column].nunique())

        print(
            df[column]
            .value_counts()
            .head(10)
        )


# --------------------------------------------------
# TARGET VARIABLE
# --------------------------------------------------

def inspect_target(df):

    print("\n========== TARGET VARIABLE ==========")

    if "Yield" not in df.columns:

        print("WARNING: Yield column not found.")
        return

    target = pd.to_numeric(
        df["Yield"],
        errors="coerce"
    )

    print(target.describe())

    print(
        "\nMissing target values:",
        target.isna().sum()
    )

    print(
        "Zero yield records:",
        (target == 0).sum()
    )

    print(
        "Negative yield records:",
        (target < 0).sum()
    )


# --------------------------------------------------
# REQUIRED COLUMN VALIDATION
# --------------------------------------------------

def validate_columns(df):

    print("\n========== COLUMN VALIDATION ==========")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print("Missing expected columns:")

        for column in missing_columns:
            print("-", column)

    else:

        print("All expected columns are available.")

    return missing_columns


# --------------------------------------------------
# SAVE INSPECTION REPORT
# --------------------------------------------------

def save_report(df, missing_columns):

    report = {

        "dataset_name": DATASET_PATH.name,

        "total_rows": int(df.shape[0]),

        "total_columns": int(df.shape[1]),

        "column_names": list(df.columns),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "missing_values": {
            column: int(value)
            for column, value
            in df.isnull().sum().items()
        },

        "duplicate_records": int(
            df.duplicated().sum()
        ),

        "missing_expected_columns": missing_columns,

        "crop_count": (
            int(df["Crop"].nunique())
            if "Crop" in df.columns else None
        ),

        "state_count": (
            int(df["State"].nunique())
            if "State" in df.columns else None
        ),

        "year_range": (
            [
                int(df["Crop_Year"].min()),
                int(df["Crop_Year"].max())
            ]
            if "Crop_Year" in df.columns else None
        )
    }

    report_path = OUTPUT_DIR / "inspection_report.json"

    with open(report_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nInspection report saved:", report_path)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n")
    print("=" * 55)
    print(" SMART CROP YIELD PREDICTION")
    print(" DATASET INSPECTION")
    print("=" * 55)

    df = load_dataset()

    inspect_shape(df)

    inspect_columns(df)

    inspect_missing_values(df)

    inspect_duplicates(df)

    inspect_statistics(df)

    inspect_categories(df)

    inspect_target(df)

    missing_columns = validate_columns(df)

    save_report(df, missing_columns)

    print("\n========================================")
    print(" DATASET INSPECTION COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()
