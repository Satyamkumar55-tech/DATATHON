import os
from pathlib import Path
import warnings
import pandas as pd


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


if __name__ == "__main__":
    df = load_data("sample_data/sales_sample.csv")
    print(df.shape)
    print(df.head(3))
    types = detect_column_types(df)
    print(types)
