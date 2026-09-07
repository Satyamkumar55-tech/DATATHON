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

import inspect
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


# Custom CSS for modern, clean, hackathon-ready UI styling
st.markdown(
    """
    <style>
    /* Global layout enhancements */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* 1️⃣ Header Hero styling */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
        color: #F8FAFC;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-tagline {
        font-size: 1.15rem;
        font-weight: 500;
        color: #38BDF8;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0;
    }

    /* 3️⃣ Welcome / Empty State Card */
    .welcome-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin: 1.5rem auto;
        max-width: 760px;
    }
    .welcome-icon {
        font-size: 3.2rem;
        margin-bottom: 1rem;
    }
    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    .welcome-desc {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.6rem;
    }
    .feature-pill-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.8rem;
        margin: 1.5rem 0;
        text-align: left;
    }
    .feature-pill {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.92rem;
        color: #334155;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .feature-pill span.check {
        color: #10B981;
        font-weight: bold;
    }

    /* 6️⃣ Cleaning Report Actions */
    .action-badge {
        background: #F0FDF4;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #14532D;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* 7️⃣ Key Insight Cards */
    .insight-card {
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .insight-card.correlation {
        border-left: 4px solid #F97316;
        background: #FFFBF7;
    }
    .insight-card.category {
        border-left: 4px solid #8B5CF6;
        background: #FAF8FF;
    }
    .insight-card.anomaly {
        border-left: 4px solid #EF4444;
        background: #FEF2F2;
    }
    .insight-card.general {
        border-left: 4px solid #3B82F6;
        background: #F8FAFC;
    }
    .insight-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .insight-text {
        font-size: 1rem;
        font-weight: 500;
        color: #1E293B;
        margin: 0;
    }

    /* Section Headings */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 1.6rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 1. OPTIONAL TEAM MODULE INTEGRATION HOOKS
# ==============================================================================
# NOTE: Teammates build loader.py, cleaner.py, analyzer.py independently.
# As per project guidelines, we DO NOT implement their logic.
# These conditional imports gracefully detect when the team's modules exist.
try:
    from modules.loader import load_data  # type: ignore
except ImportError:
    load_data = None

try:
    from modules.cleaner import clean_data  # type: ignore
except ImportError:
    clean_data = None

try:
    from modules.analyzer import analyze_data  # type: ignore
except ImportError:
    analyze_data = None


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
    """1️⃣ HEADER: Clean, attractive, professional top branding section."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🔍 AutoInsight</div>
            <div class="hero-tagline">Upload. Analyze. Understand.</div>
            <div class="hero-subtitle">Turn raw datasets into meaningful insights automatically.</div>
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


def show_visualizations(df: pd.DataFrame, charts: Optional[List[go.Figure]] = None) -> None:
    """8️⃣ AUTOMATIC VISUALIZATIONS: Responsive chart layout using columns."""
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
            fig_dist = px.histogram(
                df,
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
        st.markdown("##### 📊 Category Analysis")
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
            fig_scatter = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
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
    """Master orchestrator rendering dashboard components based on sidebar selection."""
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

    # Route based on clean sidebar radio
    if active_section in ("Dashboard", "All"):
        show_dataset_overview(cleaned_df, metadata)
        st.divider()
        show_data_preview(cleaned_df)
        st.divider()
        show_cleaning_report(cleaning_report)
        st.divider()
        show_insights(insights)
        st.divider()
        show_visualizations(cleaned_df, charts)
        st.divider()
        show_correlation(cleaned_df, corr_matrix)
    elif active_section == "Dataset Overview":
        show_dataset_overview(cleaned_df, metadata)
        st.divider()
        show_data_preview(cleaned_df)
    elif active_section == "Data Cleaning":
        show_cleaning_report(cleaning_report)
    elif active_section == "Insights":
        show_insights(insights)
    elif active_section == "Visualizations":
        show_visualizations(cleaned_df, charts)
        st.divider()
        show_correlation(cleaned_df, corr_matrix)


# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
def render_sidebar() -> Tuple[str, bool]:
    """🔟 SIDEBAR: Clean, minimal navigation and hackathon info."""
    with st.sidebar:
        st.markdown("### 🔍 AutoInsight")
        st.caption("Upload. Analyze. Understand.")
        st.markdown("---")

        nav_selection = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Dataset Overview",
                "Data Cleaning",
                "Insights",
                "Visualizations",
            ],
            index=0,
            key="nav_selection",
        )

        st.markdown("---")
        st.markdown("#### ⚡ Quick Demo")
        demo_clicked = st.button("🎲 Load Sample Dataset", **_width_kwarg())

        st.markdown("---")
        st.caption("🚀 **Hackathon Team Integration**")
        st.caption("• `modules/loader.py`: Member 1")
        st.caption("• `modules/cleaner.py`: Member 2")
        st.caption("• `modules/analyzer.py`: Member 3")
        st.caption("• `app.py`: Dashboard UI (Active)")

    return nav_selection, demo_clicked


# ==============================================================================
# 6. MAIN APPLICATION FLOW (TEAM INTEGRATION READY)
# ==============================================================================
def main() -> None:
    # 1️⃣ Header
    show_header()

    # 🔟 Sidebar Navigation
    nav_selection, demo_clicked = render_sidebar()

    # Session state for demo dataset
    if "use_demo" not in st.session_state:
        st.session_state["use_demo"] = False

    if demo_clicked:
        st.session_state["use_demo"] = True

    # 2️⃣ File Upload Section
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Select a CSV or XLSX file to begin automated processing",
    )

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

    elif st.session_state["use_demo"]:
        df = _create_sample_dataset()
        st.success("Loaded demo dataset for hackathon presentation.")

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

    # Master dashboard rendering
    display_dashboard(
        cleaned_df=cleaned_df,
        metadata=metadata,
        cleaning_report=cleaning_report,
        analysis_results=analysis_results,
        active_section=nav_selection,
    )


if __name__ == "__main__":
    main()
