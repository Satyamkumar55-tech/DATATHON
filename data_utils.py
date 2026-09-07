import os
import re
from pathlib import Path
import warnings
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Regex patterns for messy numeric formats
# ---------------------------------------------------------------------------
_RE_CURRENCY = re.compile(r"^\s*\$[\d,]+(\.[\d]+)?\s*$")            # $1,250.00
_RE_PERCENTAGE = re.compile(r"^\s*-?[\d]+(\.[\d]+)?\s*%\s*$")       # 95%  / 12.5%
_RE_COMMA_NUMBER = re.compile(r"^\s*-?\d{1,3}(,\d{3})+(\.[\d]+)?\s*$")  # 12,500

# Detection threshold: if this fraction of non-null STRING values match, convert the column
_DETECTION_THRESHOLD = 0.50


def _is_string_dtype(series):
    """True for object, StringDtype (pandas >= 1.0), and legacy 'str' dtype."""
    if pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    return False


def _try_strip_and_clean(val):
    """Strip whitespace from a scalar string value; leave non-strings unchanged."""
    if isinstance(val, str):
        return val.strip()
    return val


def _column_match_ratio(series, pattern):
    """Return the fraction of non-null STRING values that match *pattern*.
    Only string-typed cells are tested; numeric cells already converted are excluded.
    The denominator is the count of non-null string cells (not total non-null).
    """
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    str_vals = non_null[non_null.apply(lambda v: isinstance(v, str))]
    if str_vals.empty:
        return 0.0
    matched = str_vals.apply(lambda v: bool(pattern.match(v))).sum()
    return matched / len(str_vals)


def _convert_currency(val):
    """'$1,250.00' -> 1250.0; failures -> np.nan."""
    if pd.isna(val):
        return np.nan
    if not isinstance(val, str):
        return val
    cleaned = val.strip().lstrip("$").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _convert_percentage(val):
    """'95%' -> 0.95; failures -> np.nan."""
    if pd.isna(val):
        return np.nan
    if not isinstance(val, str):
        return val
    cleaned = val.strip().rstrip("%").strip()
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return np.nan


def _convert_comma_number(val):
    """'12,500' or '500' -> float; failures -> np.nan."""
    if pd.isna(val):
        return np.nan
    if not isinstance(val, str):
        # Already numeric – pass through
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
    cleaned = val.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


# Detection threshold: if ≥ 50 % of non-null values match, convert the column
_DETECTION_THRESHOLD = 0.50


def preprocess_dataframe(df):
    """
    Pre-clean raw string formatting in a DataFrame BEFORE type detection.

    Steps (in order, per column):
      1. Strip leading/trailing whitespace from every string cell.
      2. If ≥50% of non-null values look like currency  ($1,250.00) → convert to float.
      3. If ≥50% of non-null values look like percentages (95%)      → convert to float (÷100).
      4. If ≥50% of non-null values look like comma-numbers (12,500)  → convert to float.
      5. Fill missing values:
           - numeric columns  → column median
           - date columns     → column median date
           - other columns    → 'Unknown'
      6. Invalid values that cannot be parsed become NaN (handled in step 5).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    cleaned_df : pd.DataFrame   – a copy with all conversions applied
    log        : dict           – conversion counts and human-readable actions list
    """
    if df is None or df.empty:
        return (df if df is not None else pd.DataFrame()), {
            "whitespace_stripped_columns": [],
            "currency_converted": {},
            "percentage_converted": {},
            "comma_number_converted": {},
            "missing_filled": {},
            "actions": ["No operations performed: DataFrame is empty."],
        }

    cleaned_df = df.copy()
    log = {
        "whitespace_stripped_columns": [],
        "currency_converted": {},
        "percentage_converted": {},
        "comma_number_converted": {},
        "missing_filled": {},
        "actions": [],
    }

    for col in cleaned_df.columns:
        series = cleaned_df[col]

        # ------------------------------------------------------------------ #
        # Step 1 – Strip whitespace from string values
        # ------------------------------------------------------------------ #
        if _is_string_dtype(series):
            stripped = series.apply(_try_strip_and_clean)
            if not stripped.equals(series):
                cleaned_df[col] = stripped
                log["whitespace_stripped_columns"].append(col)
            series = cleaned_df[col]  # refresh reference

        # ------------------------------------------------------------------ #
        # Steps 2-4 – Detect & convert messy numeric formats (string cols only)
        # ------------------------------------------------------------------ #
        if _is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series):
            # Priority order: currency > percentage > comma-number
            if _column_match_ratio(series, _RE_CURRENCY) >= _DETECTION_THRESHOLD:
                converted = series.apply(_convert_currency)
                count = int(series.notna().sum() - converted.isna().sum()
                            + (converted.notna().sum() - series.apply(
                                lambda v: isinstance(v, (int, float)) and not pd.isna(v)
                            ).sum()))
                # simpler: count how many non-null strings were converted
                matched_count = int(series.dropna().apply(
                    lambda v: isinstance(v, str) and bool(_RE_CURRENCY.match(v))
                ).sum())
                cleaned_df[col] = converted
                log["currency_converted"][col] = matched_count
                log["actions"].append(
                    f"Converted {matched_count} currency value(s) in '{col}' to numeric"
                )

            elif _column_match_ratio(series, _RE_PERCENTAGE) >= _DETECTION_THRESHOLD:
                matched_count = int(series.dropna().apply(
                    lambda v: isinstance(v, str) and bool(_RE_PERCENTAGE.match(v))
                ).sum())
                cleaned_df[col] = series.apply(_convert_percentage)
                log["percentage_converted"][col] = matched_count
                log["actions"].append(
                    f"Converted {matched_count} percentage value(s) in '{col}' to decimal"
                )

            elif _column_match_ratio(series, _RE_COMMA_NUMBER) >= _DETECTION_THRESHOLD:
                matched_count = int(series.dropna().apply(
                    lambda v: isinstance(v, str) and bool(_RE_COMMA_NUMBER.match(v))
                ).sum())
                cleaned_df[col] = series.apply(_convert_comma_number)
                log["comma_number_converted"][col] = matched_count
                log["actions"].append(
                    f"Converted {matched_count} comma-formatted number(s) in '{col}' to numeric"
                )

    # ---------------------------------------------------------------------- #
    # Step 5 – Fill missing values
    # ---------------------------------------------------------------------- #
    for col in cleaned_df.columns:
        series = cleaned_df[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            fill_val = series.median()
            cleaned_df[col] = series.fillna(fill_val)
            log["missing_filled"][col] = missing_count
            log["actions"].append(
                f"Filled {missing_count} missing numeric value(s) in '{col}' with median ({fill_val})"
            )
        else:
            # Try date detection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_dates = pd.to_datetime(series, errors="coerce")
            non_null_dates = parsed_dates.dropna()
            date_ratio = len(non_null_dates) / len(series) if len(series) > 0 else 0

            if date_ratio > 0.8 and len(non_null_dates) > 0:
                # Median date = sort and pick the middle
                sorted_dates = non_null_dates.sort_values()
                median_date = sorted_dates.iloc[len(sorted_dates) // 2]
                cleaned_df[col] = parsed_dates.fillna(median_date)
                log["missing_filled"][col] = missing_count
                log["actions"].append(
                    f"Filled {missing_count} missing date(s) in '{col}' with median date "
                    f"({median_date.date()})"
                )
            else:
                # Categorical / text → mode or 'Unknown'
                non_null = series.dropna()
                fill_val = non_null.mode().iloc[0] if not non_null.empty else "Unknown"
                cleaned_df[col] = series.fillna(fill_val)
                log["missing_filled"][col] = missing_count
                log["actions"].append(
                    f"Filled {missing_count} missing value(s) in '{col}' with "
                    f"mode/Unknown ('{fill_val}')"
                )

    if log["whitespace_stripped_columns"]:
        log["actions"].insert(
            0,
            f"Stripped whitespace from string values in: "
            f"{log['whitespace_stripped_columns']}",
        )

    return cleaned_df, log


def load_data(file_path_or_uploadedfile):
    """Load CSV or Excel into a pandas DataFrame."""
    try:
        # Determine file name from path or file-like object (e.g., Streamlit UploadedFile)
        if hasattr(file_path_or_uploadedfile, "name"):
            filename = file_path_or_uploadedfile.name
        elif isinstance(file_path_or_uploadedfile, (str, Path)):
            filename = str(file_path_or_uploadedfile)
        else:
            filename = getattr(file_path_or_uploadedfile, "name", "")

        # Check extension
        ext = os.path.splitext(filename)[1].lower()

        # Reset pointer if it's a file-like buffer
        if hasattr(file_path_or_uploadedfile, "seek") and callable(file_path_or_uploadedfile.seek):
            file_path_or_uploadedfile.seek(0)

        if ext == ".csv":
            try:
                df = pd.read_csv(file_path_or_uploadedfile)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(file_path_or_uploadedfile)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            except ValueError as e:
                if "no sheets" in str(e).lower() or "empty" in str(e).lower():
                    df = pd.DataFrame()
                else:
                    raise
        else:
            raise ValueError(
                f"Unsupported file format '{ext or filename}'. Please provide a .csv, .xlsx, or .xls file."
            )

        # Clean column names if columns exist
        if df.columns is not None and len(df.columns) > 0:
            df.columns = [
                str(col).strip().lstrip("\ufeff").strip()
                for col in df.columns
            ]

        return df

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading file '{filename}': {e}") from e


def detect_column_types(df):
    """Classify each column as 'numeric', 'categorical', 'date', or 'text'."""
    if df is None or df.empty or len(df.columns) == 0:
        if df is not None and hasattr(df, "columns"):
            return {col: "text" for col in df.columns}
        return {}

    col_types = {}
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        # Handle all-null columns safely
        if len(non_null) == 0:
            if pd.api.types.is_numeric_dtype(series):
                col_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_types[col] = "date"
            else:
                col_types[col] = "text"
            continue

        # 1. Try pd.to_datetime(df[col], errors='coerce') — if over 80% parse, classify as 'date'
        # Skip pure numeric types as pd.to_datetime interprets raw numbers as epoch nanoseconds
        is_date = False
        if not pd.api.types.is_numeric_dtype(series):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(series, errors="coerce")
                if (parsed.dropna().count() / len(non_null)) > 0.8:
                    is_date = True
            except Exception:
                pass

        if is_date:
            col_types[col] = "date"
        # 2. Else if pd.api.types.is_numeric_dtype(df[col]), classify as 'numeric'
        elif pd.api.types.is_numeric_dtype(series):
            col_types[col] = "numeric"
        # 3. Else if df[col].nunique() < 20 or (nunique/len(df)) < 0.05, classify as 'categorical'
        elif series.nunique() < 20 or (total_rows > 0 and (series.nunique() / total_rows) < 0.05):
            col_types[col] = "categorical"
        # 4. Else classify as 'text'
        else:
            col_types[col] = "text"

    return col_types


def get_dataset_overview(df, column_types):
    """Return a summary dict of the dataset before cleaning."""
    if df is None:
        df = pd.DataFrame()

    num_rows = int(df.shape[0])
    num_cols = int(df.shape[1])
    column_names = [str(c) for c in df.columns]

    numeric_columns = [col for col, t in column_types.items() if t == "numeric"]
    categorical_columns = [col for col, t in column_types.items() if t == "categorical"]
    date_columns = [col for col, t in column_types.items() if t == "date"]
    text_columns = [col for col, t in column_types.items() if t == "text"]

    if num_rows == 0 or num_cols == 0:
        missing_values_per_column = {str(c): 0 for c in column_names}
        total_missing_values = 0
        duplicate_row_count = 0
    else:
        missing_series = df.isnull().sum()
        missing_values_per_column = {str(col): int(val) for col, val in missing_series.items()}
        total_missing_values = int(missing_series.sum())
        duplicate_row_count = int(df.duplicated().sum())

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "column_names": column_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "text_columns": text_columns,
        "missing_values_per_column": missing_values_per_column,
        "total_missing_values": total_missing_values,
        "duplicate_row_count": duplicate_row_count,
    }


if __name__ == "__main__":
    import json

    # ── Test 1: Inline messy test dataframe ─────────────────────────────────
    print("\n" + "=" * 60)
    print("MESSY TEST DATAFRAME")
    print("=" * 60)
    messy_df = pd.DataFrame({
        "Price":    ["$1,200.00", "$850.50", None],
        "Discount": ["10%", "25%", "5%"],
        "Quantity": ["1,000", "2,500", "500"],
    })
    print("Raw input:\n", messy_df)
    cleaned, log = preprocess_dataframe(messy_df)
    print("\nCleaned output:\n", cleaned)
    print("\nCleaning log:")
    print(json.dumps(log, indent=2, default=str))

    # ── Test 2-4: 3 sample files ─────────────────────────────────────────────
    sample_files = [
        "sample_data/sales_sample.csv",
        "sample_data/health_sample.csv",
        "sample_data/marketing_sample.csv",
    ]

    for file_path in sample_files:
        print(f"\n{'='*20} {file_path} {'='*20}")
        df = load_data(file_path)
        cleaned_df, log = preprocess_dataframe(df)
        print("Original shape:", df.shape, " → Cleaned shape:", cleaned_df.shape)
        types = detect_column_types(cleaned_df)
        print("Detected types:", types)
        overview = get_dataset_overview(cleaned_df, types)
        print("Overview:", overview)
        print("Cleaning actions:", log["actions"])
