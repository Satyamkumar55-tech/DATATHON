import json
from typing import Any, Dict
import pandas as pd


def _format_stat(val: Any) -> Any:
    """Helper to cleanly format numeric stats as int or rounded float."""
    if pd.isna(val):
        return None
    f_val = float(val)
    if f_val.is_integer():
        return int(f_val)
    return round(f_val, 2)


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Analyzes an already-cleaned pandas DataFrame and returns a dictionary of
    numeric/categorical summaries, correlations, outlier counts, and plain-language insights.
    """
    output: Dict[str, Any] = {
        "numeric_summary": {},
        "categorical_summary": {},
        "correlations": [],
        "outliers": {},
        "insights": []
    }

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return output

    # 1. Numeric Analysis
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            output["numeric_summary"][col] = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None
            }
        else:
            output["numeric_summary"][col] = {
                "mean": _format_stat(series.mean()),
                "median": _format_stat(series.median()),
                "min": _format_stat(series.min()),
                "max": _format_stat(series.max())
            }

    # 2. Categorical Analysis
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    for col in categorical_cols:
        series = df[col].dropna()
        unique_count = int(series.nunique())
        if series.empty:
            most_common = None
        else:
            mode_val = series.mode()
            if not mode_val.empty:
                val = mode_val.iloc[0]
                most_common = val.item() if hasattr(val, "item") else val
            else:
                most_common = None

        output["categorical_summary"][col] = {
            "most_common": most_common,
            "unique_count": unique_count
        }

    # 3. Correlation Analysis
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                corr_val = corr_matrix.loc[col_a, col_b]

                if pd.isna(corr_val):
                    continue

                if abs(corr_val) > 0.6:
                    rounded_corr = round(float(corr_val), 2)
                    output["correlations"].append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "value": rounded_corr
                    })

                    rel_type = "positive" if corr_val > 0 else "negative"
                    output["insights"].append(
                        f"🔥 {col_a} and {col_b} have a strong {rel_type} relationship."
                    )

    # 4. Outlier Detection (IQR Method)
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        if outlier_count > 0:
            output["outliers"][col] = outlier_count
            output["insights"].append(
                f"⚠️ {outlier_count} unusual values detected in {col}."
            )

    return output


def perform_clustering(df, types):
    """Run K-Means clustering on numeric columns if enough data exists."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    numeric = [c for c, t in types.items() if t == "numeric"]
    if len(numeric) < 2 or len(df) < 10:
        return {"available": False, "reason": "Need at least 2 numeric columns and 10 rows."}
    
    data = df[numeric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        return {"available": False, "reason": "Not enough complete numeric rows for clustering."}

    cols = numeric[:8]
    X = StandardScaler().fit_transform(data[cols])
    k = min(4, max(2, int(np.sqrt(len(data) / 2))))
    k = min(k, 8)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    summary = data.copy()
    summary["Cluster"] = labels
    cluster_summary = summary.groupby("Cluster")[cols].mean().round(2).reset_index()
    return {"available": True, "n_clusters": k, "summary": cluster_summary}


if __name__ == "__main__":
    df = pd.read_csv("sample.csv")
    output = analyze_dataset(df)
    import json
    print(json.dumps(output, indent=2, default=str))
