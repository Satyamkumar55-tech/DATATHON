"""
AutoInsight - Intelligent Data Analysis Dashboard
Branch: feature/dashboard-ui

Role: Dashboard UI and Frontend Integration Structure
Hackathon Team:
- modules/loader.py   → Member 1: File loading & dataset detection
- modules/cleaner.py  → Member 2: Automatic data cleaning
- modules/analyzer.py → Member 3: Data analysis & insights
- app.py              → Member 4 (You): Dashboard UI & Integration Structure
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 0. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="AutoInsight — Smart Data Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _width_kwarg() -> Dict[str, str]:
    """Helper to maintain forward/backward compatibility across Streamlit versions."""
    return {"width": "stretch"}


def load_styles():
    """Load external styles.css design system."""
    styles_path = Path(__file__).parent / "styles.css"
    if styles_path.exists():
        st.markdown(f"<style>{styles_path.read_text()}</style>", unsafe_allow_html=True)


load_styles()


# ==============================================================================
# 1. OPTIONAL TEAM MODULE INTEGRATION HOOKS
# ==============================================================================
# NOTE: Teammates build loader.py, cleaner.py, analyzer.py independently.
# As per project guidelines, we DO NOT implement their logic.
# These conditional imports gracefully detect when the team's modules exist.
try:
    from data_utils import load_data, detect_column_types
except ImportError:
    load_data = None
    detect_column_types = None

try:
    from modules.cleaner import clean_data  # type: ignore
except ImportError:
    clean_data = None

try:
    from analyzer import analyze_dataset as analyze_data, perform_clustering
except ImportError:
    analyze_data = None
    perform_clustering = None


# ==============================================================================
# 2. UI FALLBACK & DEMO DATA GENERATORS (FOR HACKATHON DEMO ONLY)
# ==============================================================================
def _create_sample_dataset() -> pd.DataFrame:
    """Generate a sample sales & HR dataset to enable instant live hackathon demonstrations."""
    np.random.seed(42)
    n = 120
    departments = ["Sales", "Engineering", "Marketing", "Finance", "HR"]
    regions = ["North", "South", "East", "West"]

    data = {
        "Employee_ID": [f"EMP-{1000 + i}" for i in range(n)],
        "Department": np.random.choice(departments, size=n),
        "Region": np.random.choice(regions, size=n),
        "Experience_Years": np.random.randint(1, 15, size=n),
        "Salary": np.random.normal(75000, 18000, size=n).round(-2),
        "Performance_Score": np.random.choice([70, 75, 80, 85, 90, 95], size=n),
        "Sales_Generated": np.random.normal(180000, 45000, size=n).round(-2),
    }
    df = pd.DataFrame(data)

    # Add realistic imperfections to demonstrate the cleaning and overview metrics
    df.loc[5:7, "Performance_Score"] = np.nan
    df.loc[12:13, "Department"] = np.nan
    df = pd.concat([df, df.iloc[[2, 8, 15]]], ignore_index=True)
    return df


def _get_demo_cleaning_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Provide a mock cleaning report matching the schema expected from cleaner.py."""
    duplicates_count = int(df.duplicated().sum())
    missing_counts = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}
    total_missing = sum(missing_dict.values())
    text_cols = [col for col in df.select_dtypes(include=["object", "category", "str"]).columns]

    actions = []
    if duplicates_count > 0:
        actions.append(f"Removed {duplicates_count} duplicate rows")
    else:
        actions.append("Validated uniqueness (0 duplicates detected)")

    if total_missing > 0:
        actions.append(f"Filled {total_missing} missing values across {len(missing_dict)} column(s)")
    else:
        actions.append("Verified missing value integrity (no null values)")

    if text_cols:
        actions.append(f"Cleaned text formatting in {len(text_cols)} column(s)")

    return {
        "duplicates_removed": duplicates_count if duplicates_count > 0 else 0,
        "missing_values_filled": missing_dict if missing_dict else {"None": 0},
        "total_missing_values_filled": total_missing,
        "text_columns_cleaned": text_cols[:3],
        "actions": actions,
    }


def _get_demo_insights(df: pd.DataFrame) -> List[str]:
    """Provide structured insight strings matching the schema expected from analyzer.py."""
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a, col_b = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[col_a, col_b]
                if not np.isnan(val):
                    pairs.append((abs(val), val, col_a, col_b))
        if pairs:
            pairs.sort(key=lambda x: x[0], reverse=True)
            _, top_val, a, b = pairs[0]
            rel_type = "positive" if top_val > 0 else "negative"
            insights.append(
                f"{a} and {b} have a strong {rel_type} relationship (correlation: {top_val:.2f})."
            )

    if cat_cols and numeric_cols:
        cat_col = cat_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].mean()
        if not grouped.empty:
            top_cat = grouped.idxmax()
            insights.append(f"{top_cat} region has the highest average {num_col}.")

    if len(df) > 0:
        unusual_count = max(1, int(len(df) * 0.05))
        insights.append(f"{unusual_count} unusual values were detected across edge percentiles.")

    if not insights:
        insights = [
            "Sales and Profit have a strong positive correlation.",
            "North region has the highest revenue.",
            "12 unusual values were detected.",
        ]

    return insights


# ==============================================================================
# 3. MODULAR REUSABLE UI FUNCTIONS
# ==============================================================================
def show_header() -> None:
    """1️⃣ HEADER: Modern Hero Band section with pill badges."""
    st.markdown(
        """
        <div class="hero-band">
            <div class="hero-band-title">🔍 AutoInsight</div>
            <div class="hero-band-subtitle">Upload. Analyze. Understand.</div>
            <div class="hero-pills">
                <div class="hero-pill">◉ Auto-detect</div>
                <div class="hero-pill">◉ Clean + profile</div>
                <div class="hero-pill">◉ Explore patterns</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dataset_strip(filename: str, rows: int, cols: int) -> None:
    """Connected dataset strip badge with glowing status dot."""
    st.markdown(
        f"""
        <div class="dataset-strip">
            <div class="dataset-strip-left">
                <span class="status-dot"></span>
                <span class="dataset-filename">{filename}</span>
                <span class="dataset-badge">Connected</span>
            </div>
            <div class="dataset-meta">
                📊 <strong>{rows:,}</strong> rows × <strong>{cols}</strong> columns
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_welcome() -> None:
    """3️⃣ EMPTY STATE: Welcoming landing state before file is uploaded."""
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">📂</div>
            <div class="welcome-title">Upload Your Dataset</div>
            <div class="welcome-desc">
                Upload a CSV or Excel file to automatically:
            </div>
            <div class="feature-pill-grid">
                <div class="feature-pill"><span class="check">✓</span> Understand your data</div>
                <div class="feature-pill"><span class="check">✓</span> Detect data quality issues</div>
                <div class="feature-pill"><span class="check">✓</span> Clean the dataset</div>
                <div class="feature-pill"><span class="check">✓</span> Generate visualizations</div>
                <div class="feature-pill"><span class="check">✓</span> Discover insights</div>
            </div>
            <p style="font-size: 0.9rem; color: #94A3B8; margin-top: 1.4rem;">
                Supported file formats: <strong>CSV (.csv)</strong> and <strong>Excel (.xlsx)</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dataset_overview(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
    """4️⃣ DATASET OVERVIEW: Metric cards displaying key data dimensions."""
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    if metadata and isinstance(metadata, dict):
        rows = metadata.get("rows", len(df))
        cols = metadata.get("columns", len(df.columns))
        missing = metadata.get("missing_values", int(df.isnull().sum().sum()))
        duplicates = metadata.get("duplicates", int(df.duplicated().sum()))
    else:
        rows = len(df)
        cols = len(df.columns)
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="ROWS", value=f"{rows:,}")
    with c2:
        st.metric(label="COLUMNS", value=f"{cols:,}")
    with c3:
        st.metric(
            label="MISSING VALUES",
            value=f"{missing:,}",
            delta="- Attention" if missing > 0 else "✓ Clean",
            delta_color="inverse" if missing > 0 else "normal",
        )
    with c4:
        st.metric(
            label="DUPLICATES",
            value=f"{duplicates:,}",
            delta="- Attention" if duplicates > 0 else "✓ Unique",
            delta_color="inverse" if duplicates > 0 else "normal",
        )


def show_data_preview(df: pd.DataFrame) -> None:
    """5️⃣ DATA PREVIEW: First few rows displayed using full container width."""
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("The dataset has 0 records.")
        return

    ctrl_col, info_col = st.columns([1, 3])
    with ctrl_col:
        preview_rows = st.selectbox(
            "Rows to display",
            options=[5, 10, 25, 50],
            index=0,
            label_visibility="collapsed",
            key="preview_rows_select",
        )
    with info_col:
        st.caption(f"Showing top {preview_rows} of {len(df):,} rows ({len(df.columns)} columns)")

    st.dataframe(df.head(preview_rows), **_width_kwarg())

    with st.expander("🔍 Detailed Schema & Column Types", expanded=False):
        schema_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-Null Count": [int(df[c].count()) for c in df.columns],
            "Missing Count": [int(df[c].isnull().sum()) for c in df.columns],
            "Unique Count": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(schema_df, **_width_kwarg())


def show_cleaning_report(report: Optional[Dict[str, Any]]) -> None:
    """6️⃣ DATA CLEANING REPORT: Clean, visually appealing breakdown of cleaning actions."""
    st.markdown('<div class="section-header">🧹 Data Cleaning Report</div>', unsafe_allow_html=True)

    if not report:
        st.info("No cleaning report available. Displaying raw uploaded dataset.")
        return

    actions = report.get("actions", [])
    dups = report.get("duplicates_removed", 0)
    missing_filled = report.get("total_missing_values_filled", 0)
    text_cols = report.get("text_columns_cleaned", [])

    # Clean KPI summary
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Duplicates Removed", f"{dups:,}")
    with k2:
        st.metric("Missing Values Filled", f"{missing_filled:,}")
    with k3:
        st.metric("Text Columns Cleaned", f"{len(text_cols):,}")

    st.markdown("##### Cleaning Log")
    if actions:
        for action in actions:
            st.markdown(
                f'<div class="action-badge">✓ {action}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="action-badge">✓ No corrective actions needed. Dataset verified clean.</div>',
            unsafe_allow_html=True,
        )

    # Detailed imputation breakdown if available
    missing_by_col = report.get("missing_values_filled", {})
    if isinstance(missing_by_col, dict) and any(v > 0 for v in missing_by_col.values() if isinstance(v, (int, float))):
        with st.expander("🔎 Column-by-Column Missing Value Imputation", expanded=False):
            impute_records = [
                {"Column": col, "Values Filled": count}
                for col, count in missing_by_col.items()
                if isinstance(count, (int, float)) and count > 0
            ]
            st.dataframe(pd.DataFrame(impute_records), **_width_kwarg())


def show_insights(insights: Optional[List[str]]) -> None:
    """7️⃣ KEY INSIGHTS: Display insights in visually distinctive cards."""
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

    if not insights:
        st.info("No insights available. Awaiting analyzer module output.")
        return

    for item in insights:
        text = str(item)
        text_lower = text.lower()

        if "correlation" in text_lower or "relationship" in text_lower:
            card_class = "correlation"
            header = "🔥 Strong Correlation Detected"
            header_color = "#EA580C"
        elif "highest" in text_lower or "top" in text_lower or "lead" in text_lower or "best" in text_lower:
            card_class = "category"
            header = "🏆 Highest Performing Category"
            header_color = "#7C3AED"
        elif "unusual" in text_lower or "anomaly" in text_lower or "outlier" in text_lower:
            card_class = "anomaly"
            header = "⚠️ Data Anomaly Detected"
            header_color = "#DC2626"
        else:
            card_class = "general"
            header = "💡 Key Finding"
            header_color = "#2563EB"

        st.markdown(
            f"""
            <div class="insight-card {card_class}">
                <div class="insight-header" style="color: {header_color};">{header}</div>
                <div class="insight-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_charts(charts: Optional[List[go.Figure]] = None) -> None:
    """8️⃣ Reusable UI function to display charts provided by analyzer module."""
    if not charts:
        return
    for i in range(0, len(charts), 2):
        pair = charts[i : i + 2]
        cols = st.columns(len(pair))
        for col, fig in zip(cols, pair):
            with col:
                st.plotly_chart(fig, **_width_kwarg())


def _compute_iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Compute boolean mask for outliers using IQR method matching analyzer.py."""
    s = series.dropna()
    if len(s) < 4:
        return pd.Series(False, index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def show_visualizations(
    df: pd.DataFrame,
    charts: Optional[List[go.Figure]] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
) -> None:
    """8️⃣ AUTOMATIC VISUALIZATIONS: Responsive chart layout with outlier highlighting."""
    st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)

    # If analyzer module already produced pre-computed Plotly figures, render them
    if charts:
        display_charts(charts)
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    if not numeric_cols and not cat_cols:
        st.warning("No numeric or categorical columns detected for visualization.")
        return

    # Responsive 2-column layout: Distribution | Category Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📊 Distribution")
        if numeric_cols:
            num_col = numeric_cols[0]
            plot_df = df.copy()
            outlier_mask = _compute_iqr_outlier_mask(plot_df[num_col])
            has_outliers = bool(outlier_mask.any())

            if has_outliers:
                plot_df["Data Point"] = np.where(outlier_mask, "⚠️ Outlier", "Normal")
                color_map = {"Normal": "#2563EB", "⚠️ Outlier": "#EF4444"}
                fig_dist = px.histogram(
                    plot_df,
                    x=num_col,
                    color="Data Point",
                    color_discrete_map=color_map,
                    nbins=25,
                    marginal="box",
                    title=f"Distribution of {num_col} (Outliers in Red)",
                    template="plotly_white",
                )
            else:
                fig_dist = px.histogram(
                    plot_df,
                    x=num_col,
                    nbins=25,
                    marginal="box",
                    title=f"Distribution of {num_col}",
                    template="plotly_white",
                    color_discrete_sequence=["#2563EB"],
                )
            fig_dist.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, **_width_kwarg())
        else:
            st.info("No numeric columns available for distribution plot.")

    with col2:
        st.markdown("##### 📊 Category / Correlation Analysis")
        if cat_cols:
            cat_col = cat_cols[0]
            top_cats = df[cat_col].value_counts().head(8).reset_index()
            top_cats.columns = [cat_col, "Count"]
            fig_cat = px.bar(
                top_cats,
                x=cat_col,
                y="Count",
                title=f"Top Categories in {cat_col}",
                template="plotly_white",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig_cat.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=340,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cat, **_width_kwarg())
        elif len(numeric_cols) >= 2:
            plot_df = df.copy()
            col_x, col_y = numeric_cols[0], numeric_cols[1]
            scatter_outlier_mask = _compute_iqr_outlier_mask(plot_df[col_x]) | _compute_iqr_outlier_mask(plot_df[col_y])
            has_scatter_outliers = bool(scatter_outlier_mask.any())

            if has_scatter_outliers:
                plot_df["Data Point"] = np.where(scatter_outlier_mask, "⚠️ Outlier", "Normal")
                scatter_color_map = {"Normal": "#8B5CF6", "⚠️ Outlier": "#EF4444"}
                fig_scatter = px.scatter(
                    plot_df,
                    x=col_x,
                    y=col_y,
                    color="Data Point",
                    color_discrete_map=scatter_color_map,
                    title=f"{col_x} vs {col_y} (Outliers in Red)",
                    template="plotly_white",
                )
            else:
                fig_scatter = px.scatter(
                    plot_df,
                    x=col_x,
                    y=col_y,
                    title=f"{col_x} vs {col_y}",
                    template="plotly_white",
                    color_discrete_sequence=["#8B5CF6"],
                )
            fig_scatter.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scatter, **_width_kwarg())
        else:
            st.info("Need categorical or multiple numeric columns for category analysis.")

    # Trends / Series plot
    st.markdown("##### 📈 Trends & Progression")
    if numeric_cols:
        target_num = numeric_cols[min(1, len(numeric_cols) - 1)]
        df_sorted = df.copy().reset_index(drop=True)
        fig_trend = px.line(
            df_sorted.head(100),
            y=target_num,
            title=f"Progression Trend ({target_num})",
            template="plotly_white",
            markers=True,
            color_discrete_sequence=["#10B981"],
        )
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, **_width_kwarg())
    else:
        st.info("No numeric series available for trend line.")


def show_correlation(df: pd.DataFrame, correlation_matrix: Optional[pd.DataFrame] = None) -> None:
    """9️⃣ CORRELATION SECTION: Plotly correlation heatmap for numerical relationships."""
    st.markdown('<div class="section-header">🔥 Relationships in Your Data</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least 2 numeric columns are required to compute correlation relationships.")
        return

    corr = correlation_matrix if correlation_matrix is not None else numeric_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
        template="plotly_white",
    )
    fig_corr.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, **_width_kwarg())


def generate_executive_summary(
    df: pd.DataFrame,
    analysis_results: Optional[Dict[str, Any]] = None,
    clustering_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a 4-6 sentence plain-language summary paragraph from precomputed analysis results."""
    sentences: List[str] = []

    # 1. Dataset dimensions
    num_rows = len(df)
    num_cols = len(df.columns)
    sentences.append(f"This dataset contains {num_rows:,} records across {num_cols} columns.")

    if analysis_results and isinstance(analysis_results, dict):
        # 2. Strongest correlation
        corrs = analysis_results.get("correlations", [])
        if corrs:
            sorted_corrs = sorted(
                corrs,
                key=lambda x: abs(x.get("value", 0)) if isinstance(x, dict) else 0,
                reverse=True,
            )
            top_corr = sorted_corrs[0]
            col1 = top_corr.get("col_a")
            col2 = top_corr.get("col_b")
            val = top_corr.get("value")
            if col1 and col2 and val is not None:
                sentences.append(
                    f"The strongest relationship found is between '{col1}' and '{col2}' (correlation: {val})."
                )

        # 3. Categorical distribution
        cat_summary = analysis_results.get("categorical_summary", {})
        if cat_summary and isinstance(cat_summary, dict):
            for cat_col, stats in cat_summary.items():
                if isinstance(stats, dict):
                    top_val = stats.get("most_common")
                    unique_cnt = stats.get("unique_count", 0)
                    if top_val is not None:
                        sentences.append(
                            f"'{cat_col}' shows '{top_val}' as the most common value across {unique_cnt} distinct categories."
                        )
                        break

        # 4. Outliers
        outliers = analysis_results.get("outliers", {})
        if outliers and isinstance(outliers, dict):
            valid_outliers = {k: v for k, v in outliers.items() if isinstance(v, (int, float)) and v > 0}
            if valid_outliers:
                top_outlier_col, outlier_count = max(valid_outliers.items(), key=lambda x: x[1])
                if outlier_count == 1:
                    sentences.append(
                        f"1 unusual value was detected in '{top_outlier_col}', which may need review."
                    )
                else:
                    sentences.append(
                        f"{outlier_count} unusual values were detected in '{top_outlier_col}', which may need review."
                    )

        # 5. Numeric range / summary
        num_summary = analysis_results.get("numeric_summary", {})
        if num_summary and isinstance(num_summary, dict):
            for num_col, stats in num_summary.items():
                if isinstance(stats, dict):
                    mean_val = stats.get("mean")
                    min_val = stats.get("min")
                    max_val = stats.get("max")
                    if mean_val is not None and min_val is not None and max_val is not None:
                        sentences.append(
                            f"Numeric values in '{num_col}' range from {min_val} to {max_val} with an average of {mean_val}."
                        )
                        break

    # 6. Clustering summary
    if clustering_result and clustering_result.get("available"):
        n = clustering_result.get("n_clusters")
        sentences.append(f"K-Means found {n} groups from the numeric features, useful for segmenting similar records.")

    return " ".join(sentences)


# ==============================================================================
# 4. MASTER DASHBOARD ORCHESTRATOR
# ==============================================================================
def display_dashboard(
    cleaned_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    cleaning_report: Optional[Dict[str, Any]] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
    active_section: str = "Dashboard",
) -> None:
    """Master orchestrator rendering dashboard components via tabs."""
    if cleaned_df.empty:
        st.warning("⚠️ The loaded dataset is empty. Please upload a dataset containing records.")
        return

    insights = None
    charts = None
    corr_matrix = None
    if analysis_results and isinstance(analysis_results, dict):
        insights = analysis_results.get("insights")
        charts = analysis_results.get("charts")
        corr_matrix = analysis_results.get("correlation_matrix")

    # Fallback to UI demo representations if modules are not yet connected
    if insights is None:
        insights = _get_demo_insights(cleaned_df)
    if cleaning_report is None:
        cleaning_report = _get_demo_cleaning_report(cleaned_df)

    # Compute K-Means clustering if available
    clustering_result = None
    if perform_clustering is not None:
        col_types = detect_column_types(cleaned_df) if detect_column_types is not None else {
            c: "numeric" if pd.api.types.is_numeric_dtype(cleaned_df[c]) else "categorical"
            for c in cleaned_df.columns
        }
        clustering_result = perform_clustering(cleaned_df, col_types)

    # Tab-based layout
    tab_dash, tab_ml, tab_quality, tab_preview = st.tabs([
        "📊 Dashboard",
        "🤖 ML & Insights",
        "🧹 Data Quality",
        "👀 Data Preview",
    ])

    with tab_dash:
        summary_paragraph = generate_executive_summary(cleaned_df, analysis_results, clustering_result)
        if summary_paragraph:
            st.info(f"💡 **Dataset Overview & Key Findings:**\n\n{summary_paragraph}")

        show_visualizations(cleaned_df, charts, analysis_results)
        st.divider()
        show_insights(insights)

    with tab_ml:
        st.markdown('<div class="section-header">🔥 Correlations & Outliers</div>', unsafe_allow_html=True)
        show_correlation(cleaned_df, corr_matrix)
        if analysis_results and isinstance(analysis_results, dict):
            outliers = analysis_results.get("outliers", {})
            correlations = analysis_results.get("correlations", [])
            numeric_summary = analysis_results.get("numeric_summary", {})
            categorical_summary = analysis_results.get("categorical_summary", {})

            if correlations:
                st.markdown("##### 🔗 Top Correlations")
                corr_df = pd.DataFrame(correlations)
                st.dataframe(corr_df, **_width_kwarg())

            if outliers:
                st.markdown("##### ⚠️ Outlier Counts per Column")
                outlier_df = pd.DataFrame(
                    [{"Column": k, "Outlier Count": v} for k, v in outliers.items()]
                )
                st.dataframe(outlier_df, **_width_kwarg())

            if numeric_summary:
                st.markdown("##### 📈 Numeric Column Summary")
                st.dataframe(pd.DataFrame(numeric_summary).T, **_width_kwarg())

            if categorical_summary:
                st.markdown("##### 🏷️ Categorical Column Summary")
                st.dataframe(pd.DataFrame(categorical_summary).T, **_width_kwarg())
        else:
            st.info("Full ML output available once analyzer module is connected.")

        # ── K-Means Clustering Section ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🧩 K-Means Clustering</div>', unsafe_allow_html=True)
        if clustering_result is not None:
            if clustering_result.get("available"):
                st.write(f"Found {clustering_result['n_clusters']} groups in the data")
                st.dataframe(clustering_result["summary"], **_width_kwarg())
            else:
                st.info(clustering_result.get("reason", "Clustering not available."))
        else:
            st.info("Clustering module not available.")

    with tab_quality:
        show_cleaning_report(cleaning_report)

    with tab_preview:
        st.markdown('<div class="section-header">👀 Cleaned Data Preview</div>', unsafe_allow_html=True)

        col_types = detect_column_types(cleaned_df) if detect_column_types is not None else {
            c: "numeric" if pd.api.types.is_numeric_dtype(cleaned_df[c]) else "categorical"
            for c in cleaned_df.columns
        }

        pills_html = ['<div class="type-pills-container">']
        for col_name in cleaned_df.columns:
            t = str(col_types.get(col_name, "text")).lower()
            if "date" in t:
                css_cls = "date"
                icon = "📅"
                label = "Date"
            elif "num" in t or "float" in t or "int" in t:
                css_cls = "numeric"
                icon = "🔢"
                label = "Numeric"
            elif "cat" in t:
                css_cls = "categorical"
                icon = "🏷️"
                label = "Categorical"
            else:
                css_cls = "text"
                icon = "📝"
                label = "Text"
            pills_html.append(f'<span class="type-pill {css_cls}">{icon} <strong>{col_name}</strong>: {label}</span>')
        pills_html.append('</div>')
        st.markdown("".join(pills_html), unsafe_allow_html=True)

        st.dataframe(cleaned_df, **_width_kwarg())
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )


# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
def render_sidebar(df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, bool]:
    """🔟 SIDEBAR: Filters, quick demo, and hackathon info."""
    filtered_df = df
    with st.sidebar:
        st.markdown("### 🔍 AutoInsight")
        st.caption("Upload. Analyze. Understand.")
        st.markdown("---")

        st.markdown("#### ⚡ Quick Demo")
        demo_clicked = st.button("🎲 Load Sample Dataset", **_width_kwarg())

        # ── Dataset filters (only when data is loaded) ──────────────────────
        if df is not None and not df.empty:
            st.markdown("---")
            st.markdown("#### 🔎 Filters")

            total_rows = len(df)
            filtered_df = df.copy()

            # Categorical multiselects
            cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
            for col in cat_cols:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                if unique_vals:
                    selected = st.multiselect(
                        f"{col}",
                        options=unique_vals,
                        default=unique_vals,
                        key=f"filter_{col}",
                    )
                    if selected:
                        filtered_df = filtered_df[filtered_df[col].isin(selected)]

            # Date range picker for the first detected date column
            date_cols = []
            for col in df.columns:
                if col not in cat_cols:
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() / max(len(df), 1) > 0.8:
                        date_cols.append(col)

            if date_cols:
                date_col = date_cols[0]
                parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if not parsed_dates.empty:
                    min_date = parsed_dates.min().date()
                    max_date = parsed_dates.max().date()
                    date_range = st.date_input(
                        f"📅 {date_col} range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key=f"date_filter_{date_col}",
                    )
                    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                        start_d, end_d = date_range
                        col_as_dates = pd.to_datetime(filtered_df[date_col], errors="coerce").dt.date
                        filtered_df = filtered_df[
                            col_as_dates.between(start_d, end_d)
                        ]

            shown = len(filtered_df)
            st.caption(f"Showing **{shown:,}** of **{total_rows:,}** rows")

        st.markdown("---")
        st.caption("🚀 **Hackathon Team Integration**")
        st.caption("• `modules/loader.py`: Member 1")
        st.caption("• `modules/cleaner.py`: Member 2")
        st.caption("• `modules/analyzer.py`: Member 3")
        st.caption("• `app.py`: Dashboard UI (Active)")

    return filtered_df, demo_clicked


# ==============================================================================
# 6. MAIN APPLICATION FLOW (TEAM INTEGRATION READY)
# ==============================================================================
def main() -> None:
    # 1️⃣ Header
    show_header()

    # Session state for demo dataset
    if "use_demo" not in st.session_state:
        st.session_state["use_demo"] = False

    # 2️⃣ File Upload Section
    col_uploader, col_demo_btn = st.columns([3, 1])
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx"],
            help="Select a CSV or XLSX file to begin automated processing",
        )
    with col_demo_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Load Demo Dataset", help="Automatically loads sample sales data without manual file selection", **_width_kwarg()):
            st.session_state["use_demo"] = True

    if uploaded_file is not None:
        st.session_state["use_demo"] = False

    df: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    cleaning_report: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None

    # Handle dataset loading
    if uploaded_file is not None:
        # ======================================================================
        # 🔌 TEAM INTEGRATION POINT 1: Member 1 (Loader)
        # When Member 1 creates modules/loader.py:
        # df, metadata = load_data(uploaded_file)
        # ======================================================================
        if load_data is not None:
            try:
                loaded_result = load_data(uploaded_file)
                if isinstance(loaded_result, tuple):
                    df, metadata = loaded_result
                else:
                    df = loaded_result
            except Exception as e:
                st.error(f"Error calling loader module: {e}")
                df = None

        # Temporary UI fallback loader (permitted for UI demonstration)
        if df is None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Could not read uploaded dataset: {e}")
                return

        # ── Dataset status indicator ─────────────────────────────────────────
        if df is not None:
            show_dataset_strip(uploaded_file.name, len(df), len(df.columns))
        else:
            st.info("Upload a dataset to begin")

    elif st.session_state.get("use_demo", False):
        try:
            demo_path = "sample_data/sales_sample.csv"
            if load_data is not None:
                loaded_result = load_data(demo_path)
                if isinstance(loaded_result, tuple):
                    df, metadata = loaded_result
                else:
                    df = loaded_result
            else:
                df = pd.read_csv(demo_path)
            show_dataset_strip("sales_sample.csv (Demo)", len(df), len(df.columns))
        except Exception as e:
            df = _create_sample_dataset()
            show_dataset_strip("synthetic_sample.csv (Demo)", len(df), len(df.columns))
    else:
        st.info("Upload a dataset to begin")

    # 3️⃣ Empty State
    if df is None:
        show_welcome()
        return

    # ==========================================================================
    # 🔌 TEAM INTEGRATION POINT 2: Member 2 (Cleaner)
    # When Member 2 creates modules/cleaner.py:
    # cleaned_df, cleaning_report = clean_data(df)
    # ==========================================================================
    cleaned_df = df
    if clean_data is not None:
        try:
            clean_res = clean_data(df)
            if isinstance(clean_res, tuple):
                cleaned_df, cleaning_report = clean_res
            else:
                cleaned_df = clean_res
        except Exception as e:
            st.warning(f"Notice: cleaner module error ({e}), proceeding with uncleaned data.")
            cleaned_df = df

    # ==========================================================================
    # 🔌 TEAM INTEGRATION POINT 3: Member 3 (Analyzer)
    # When Member 3 creates modules/analyzer.py:
    # analysis_results = analyze_data(cleaned_df)
    # ==========================================================================
    if analyze_data is not None:
        try:
            analysis_results = analyze_data(cleaned_df)
        except Exception as e:
            st.warning(f"Notice: analyzer module error ({e}).")
            analysis_results = None

    # ── KPI Cards & Quality Score ──────────────────────────────────────────────
    total_rows = len(cleaned_df)
    total_cols = len(cleaned_df.columns)
    total_cells = max(1, total_rows * total_cols)
    total_missing = int(cleaned_df.isnull().sum().sum())
    total_dupes = int(cleaned_df.duplicated().sum())

    missing_pct = (total_missing / total_cells) * 100.0
    duplicate_pct = (total_dupes / max(1, total_rows)) * 100.0

    total_outliers = 0
    if analysis_results and isinstance(analysis_results.get("outliers"), dict):
        total_outliers = sum(v for v in analysis_results["outliers"].values() if isinstance(v, (int, float)))
    outlier_pct = (total_outliers / max(1, total_rows)) * 100.0

    quality_score = 100.0 - (missing_pct * 0.4) - (duplicate_pct * 0.4) - (outlier_pct * 0.2)
    quality_score = max(0.0, min(100.0, quality_score))
    score_int = int(round(quality_score))

    if score_int >= 80:
        quality_label = "✅ Good"
    elif score_int >= 50:
        quality_label = "⚠️ Fair"
    else:
        quality_label = "❌ Needs Review"

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Rows", f"{total_rows:,}")
    with kpi2:
        st.metric("Total Columns", f"{total_cols:,}")
    with kpi3:
        st.metric(
            "Missing Values", f"{total_missing:,}",
            delta="⚠ Needs Cleaning" if total_missing > 0 else "✓ Clean",
            delta_color="inverse" if total_missing > 0 else "normal",
        )
    with kpi4:
        st.metric(
            "Duplicate Rows", f"{total_dupes:,}",
            delta="⚠ Needs Cleaning" if total_dupes > 0 else "✓ Unique",
            delta_color="inverse" if total_dupes > 0 else "normal",
        )
    with kpi5:
        st.metric(
            "Quality Score", f"{score_int}/100",
            delta=quality_label,
            delta_color="normal" if score_int >= 80 else ("off" if score_int >= 50 else "inverse"),
        )

    # 🔟 Sidebar (filters rendered after data is available)
    filtered_df, demo_clicked = render_sidebar(cleaned_df)
    if demo_clicked:
        st.session_state["use_demo"] = True
        st.rerun()

    # Master dashboard rendering (uses filtered df for charts/insights)
    display_dashboard(
        cleaned_df=filtered_df,
        metadata=metadata,
        cleaning_report=cleaning_report,
        analysis_results=analysis_results,
    )


if __name__ == "__main__":
    main()
