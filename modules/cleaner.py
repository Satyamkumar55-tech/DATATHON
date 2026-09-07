"""
DataLens AI - Data Cleaning Module

This module provides automated data cleaning functionality for the DataLens AI project.
It handles duplicate removal, missing value imputation (median for numeric columns,
'Unknown' for categorical text columns), text whitespace normalization, and
preserves original column data types.

Main Function:
    clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a Pandas DataFrame and generates a structured report of cleaning operations.

    Operations performed:
    1. Removes duplicate rows.
    2. Fills missing values (median for numeric columns, 'Unknown' for text/categorical columns).
    3. Normalizes text formatting (strips leading/trailing whitespace and collapses multiple spaces).
    4. Preserves column data types.

    Parameters:
        df (pd.DataFrame): Input Pandas DataFrame to clean.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - cleaned_df: The cleaned Pandas DataFrame.
            - cleaning_report: Structured summary dictionary of all cleaning actions.
    """
    # Defensive check: ensure input is a Pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")

    # Work on a copy to avoid side-effects on original dataset
    cleaned_df = df.copy()
    actions: List[str] = []

    # Handle empty DataFrame safely
    if cleaned_df.empty or len(cleaned_df.columns) == 0:
        report: Dict[str, Any] = {
            "duplicates_removed": 0,
            "missing_values_filled": {},
            "total_missing_values_filled": 0,
            "text_columns_cleaned": [],
            "actions": ["No operations performed: DataFrame is empty."]
        }
        return cleaned_df, report

    # ------------------------------------------------------------------
    # 1. Remove Duplicate Rows
    # ------------------------------------------------------------------
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(cleaned_df)

    if duplicates_removed > 0:
        actions.append(f"Removed {duplicates_removed} duplicate row{'s' if duplicates_removed != 1 else ''}")

    # ------------------------------------------------------------------
    # 2. Handle Missing Values
    # ------------------------------------------------------------------
    missing_values_filled: Dict[str, int] = {}

    # Separate numeric and non-numeric columns while preserving dtypes
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns.tolist()

    # Fill missing values in numeric columns with the median
    for col in numeric_cols:
        null_count = int(cleaned_df[col].isnull().sum())
        if null_count > 0:
            median_val = cleaned_df[col].median()
            # Handle case where median is valid (not NaN, i.e., at least 1 non-null value exists)
            if pd.notna(median_val):
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                missing_values_filled[col] = null_count

    # Fill missing values in categorical/text columns with "Unknown"
    for col in non_numeric_cols:
        null_count = int(cleaned_df[col].isnull().sum())
        if null_count > 0:
            if isinstance(cleaned_df[col].dtype, pd.CategoricalDtype):
                if "Unknown" not in cleaned_df[col].cat.categories:
                    cleaned_df[col] = cleaned_df[col].cat.add_categories(["Unknown"])
            cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            missing_values_filled[col] = null_count

    total_missing_values_filled = sum(missing_values_filled.values())
    if total_missing_values_filled > 0:
        col_count = len(missing_values_filled)
        actions.append(
            f"Filled {total_missing_values_filled} missing value{'s' if total_missing_values_filled != 1 else ''} "
            f"across {col_count} column{'s' if col_count != 1 else ''}"
        )

    # ------------------------------------------------------------------
    # 3. Clean Text Columns
    # ------------------------------------------------------------------
    text_columns_cleaned: List[str] = []

    for col in non_numeric_cols:
        # Check if column contains string or object data
        if pd.api.types.is_string_dtype(cleaned_df[col]) or pd.api.types.is_object_dtype(cleaned_df[col]):
            # Perform whitespace stripping and internal space normalization
            original_series = cleaned_df[col].astype(str)
            cleaned_series = original_series.str.strip().str.replace(r'\s+', ' ', regex=True)

            # Record column if formatting changes were made
            if not original_series.equals(cleaned_series):
                text_columns_cleaned.append(col)
                cleaned_df[col] = cleaned_series

    if text_columns_cleaned:
        actions.append(
            f"Cleaned text formatting in {len(text_columns_cleaned)} column{'s' if len(text_columns_cleaned) != 1 else ''}"
        )

    if not actions:
        actions.append("Dataset is already clean; no changes were required.")

    # ------------------------------------------------------------------
    # 4. Construct Structured Cleaning Report
    # ------------------------------------------------------------------
    cleaning_report: Dict[str, Any] = {
        "duplicates_removed": duplicates_removed,
        "missing_values_filled": missing_values_filled,
        "total_missing_values_filled": total_missing_values_filled,
        "text_columns_cleaned": text_columns_cleaned,
        "actions": actions
    }

    return cleaned_df, cleaning_report


if __name__ == "__main__":
    # Quick self-test demonstration
    sample_data = {
        "Name": [" Alice ", "Bob  Smith", " Alice ", None, " Charlie "],
        "Age": [20.0, 22.0, 20.0, None, 25.0],
        "City": [" Mumbai ", "New   Delhi", " Mumbai ", "Chennai", "Bangalore"],
        "Department": ["IT", "HR", "IT", None, "Finance  Dept"]
    }

    raw_df = pd.DataFrame(sample_data)
    print("--- RAW DATAFRAME ---")
    print(raw_df)

    cleaned_df, report = clean_data(raw_df)

    print("\n--- CLEANED DATAFRAME ---")
    print(cleaned_df)

    print("\n--- CLEANING REPORT ---")
    import json
    print(json.dumps(report, indent=4))
