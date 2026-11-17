"""
FinSecure - Credit Card Fraud Detection Platform
A lightweight, professional, and accessible fraud surveillance application
engineered for final-year B.Tech CSE Project-Based Learning (PBL).

Built on the authentic Kaggle MLG-ULB European Credit Card Fraud Dataset (284,807 transactions).
Supports 4 Machine Learning and Deep Learning architectures:
- Random Forest (Ensemble Supervised ML)
- Logistic Regression (Linear Supervised ML)
- Artificial Neural Network (Supervised Deep Learning)
- Autoencoder (Unsupervised Deep Learning Anomaly Detection)
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Root setup for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prediction import FraudPredictionEngine
from src.data_processing import load_raw_dataset

# =====================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="FinSecure | Fraud Detection Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 2. FINTECH DESIGN SYSTEM (CLEAN, PROFESSIONAL, HIGH CONTRAST)
# =====================================================================
FINTECH_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Transparent Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
        z-index: 99990 !important;
        pointer-events: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }

    /* Permanently remove the collapse button so the sidebar remains locked open */
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Professional Dark Navy Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #CBD5E1 !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Modern Portal Topbar */
    .portal-topbar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .portal-title-box {
        flex: 1 1 320px;
        min-width: 260px;
    }
    .portal-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        line-height: 1.25;
        letter-spacing: -0.02em;
    }
    .portal-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin: 4px 0 0 0;
        line-height: 1.4;
    }
    .portal-badge {
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }

    /* Container Box Overrides */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #64748B !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin: 0 0 4px 0 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #0F172A !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }

    /* Section Headings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 14px;
    }

    /* Chart Explanation Box */
    .chart-explanation {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 6px;
        padding: 9px 13px;
        margin-top: 10px;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.45;
    }

    /* Multi-Model Consensus Cards */
    .consensus-metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        text-align: center;
    }
    .consensus-metric-title {
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .consensus-metric-val {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }

    /* Form Controls */
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        padding: 10px 18px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }
</style>
"""
st.markdown(FINTECH_THEME, unsafe_allow_html=True)

# Ensure the sidebar stays open on load
components.html(
    """
    <script>
    function forceExpandSidebar() {
        try {
            const pDoc = window.parent.document;
            if (!pDoc) return;
            const sidebar = pDoc.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                const isCollapsed = sidebar.getAttribute('aria-expanded') === 'false' || 
                                    sidebar.getBoundingClientRect().width < 50;
                if (isCollapsed) {
                    const btn = pDoc.querySelector('[data-testid="stSidebarCollapseButton"]') ||
                                pDoc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                                pDoc.querySelector('[data-testid="collapsedControl"] button') ||
                                pDoc.querySelector('header button');
                    if (btn) btn.click();
                }
            }
        } catch(e) {}
    }
    forceExpandSidebar();
    setTimeout(forceExpandSidebar, 200);
    setTimeout(forceExpandSidebar, 600);
    </script>
    """,
    height=0,
    width=0
)

# =====================================================================
# 3. CACHED DATA & INFERENCE ACCESS
# =====================================================================
@st.cache_resource
def get_prediction_engine():
    return FraudPredictionEngine()

@st.cache_data
def get_cached_sample_data(nrows=5000):
    try:
        return load_raw_dataset(nrows=nrows)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_cached_hourly_activity():
    p = config.HOURLY_ACTIVITY_CSV
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def get_cached_test_sample_pool():
    p = config.TEST_SAMPLE_POOL_CSV
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def get_cached_batch_template():
    p = getattr(config, "SAMPLE_BATCH_TEMPLATE_CSV", config.METRICS_DIR / "sample_batch_template.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def apply_fintech_chart_theme(fig, height=280, title=None):
    """Applies a crisp, high-contrast light theme to all Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        title=dict(
            text=title if title else "",
            font=dict(size=13, color="#0F172A", family="Plus Jakarta Sans, sans-serif")
        ) if title else None,
        font=dict(color="#0F172A", family="Plus Jakarta Sans, sans-serif", size=12),
        margin=dict(l=20, r=20, t=32 if title else 16, b=20),
        legend=dict(
            font=dict(color="#0F172A", size=11),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#0F172A", size=12),
        showgrid=False,
        linecolor="#CBD5E1"
    )
    fig.update_yaxes(
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#0F172A", size=12),
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1"
    )
    return fig

# Initialize services and datasets
engine = get_prediction_engine()
sample_df = get_cached_sample_data(nrows=5000)
hourly_df = get_cached_hourly_activity()
test_pool_df = get_cached_test_sample_pool()
batch_template_df = get_cached_batch_template()

# =====================================================================
# 4. SIDEBAR - BRANDING, 5-PAGE NAVIGATION & CURRENCY
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 6px 0 16px 0;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='background: #2563EB; color: #FFFFFF; font-weight: 800; font-size: 0.95rem; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; letter-spacing: 0.05em;'>FS</div>
            <div>
                <h3 style='color:#FFFFFF !important; margin:0; font-size:1.25rem; font-weight:800;'>FinSecure</h3>
                <p style='color:#93C5FD !important; font-size:0.75rem; margin:0;'>Fraud Detection Platform</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Exactly 5 Core Navigation Sections
    menu_selection = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Detect Fraud",
            "Batch Detection",
            "EDA & Model Performance",
            "About"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    # Currency Display Selection (Default: INR for Indian Demonstration)
    st.markdown("<p style='font-size:0.72rem; color:#94A3B8; text-transform:uppercase; font-weight:700; margin-bottom:4px;'>DISPLAY CURRENCY</p>", unsafe_allow_html=True)
    currency_choice = st.radio(
        "Display Currency",
        ["INR (₹) [Estimated Conversion]", "EUR (€) [Original Dataset]"],
        index=0,
        label_visibility="collapsed"
    )
    is_inr = "INR" in currency_choice

    if is_inr:
        st.caption("Estimated conversion: 1 EUR = ₹90.00. Model inference strictly processes original European EUR feature values.")
    else:
        st.caption("Original dataset currency from European cardholders (Sept 2013).")

    st.divider()
    st.markdown("<p style='font-size:0.72rem; color:#94A3B8; text-transform:uppercase; font-weight:700; margin-bottom:4px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#34D399 !important; font-weight:700; font-size:0.82rem; margin:0;'>● 4 Models Ready</p>", unsafe_allow_html=True)
    st.caption("Random Forest, Neural Network (ANN), Logistic Regression, Autoencoder.")

# Currency formatting helpers
def fmt_money(amount_eur: float) -> str:
    if is_inr:
        amount_inr = amount_eur * config.EUR_TO_INR_RATE
        return f"₹{amount_inr:,.2f} (€{amount_eur:,.2f})"
    return f"€{amount_eur:,.2f}"

def fmt_money_short(amount_eur: float) -> str:
    if is_inr:
        return f"₹{amount_eur * config.EUR_TO_INR_RATE:,.2f}"
    return f"€{amount_eur:,.2f}"


# =====================================================================
# PAGE 1: DASHBOARD - SIMPLE AND INFORMATIVE
# =====================================================================
if menu_selection == "Dashboard":
    st.markdown("""
    <div class='portal-topbar'>
        <div class='portal-title-box'>
            <h2 class='portal-title'>Dataset Overview</h2>
            <p class='portal-sub'>Historical transaction overview across 284,807 European cardholder records recorded over a 48-hour timeline (September 2013)</p>
        </div>
        <div class='portal-badge'>
            ● 284,807 HISTORICAL RECORDS
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Main Statistics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Total Transactions",
            value="284,807",
            delta="100% Historical Data"
        )
    with k2:
        st.metric(
            label="Legitimate Transactions",
            value="284,315",
            delta="99.828% of Dataset"
        )
    with k3:
        st.metric(
            label="Fraudulent Transactions",
            value="492",
            delta="0.172% Total Cases",
            delta_color="inverse"
        )
    with k4:
        st.metric(
            label="Fraud Percentage",
            value="0.172%",
            delta="1 fraud in ~580 txns",
            delta_color="inverse"
        )

    st.write("")

    # Visualizations: Row 1 - Class Imbalance
    with st.container(border=True):
        st.markdown("<div class='section-title'>Class Distribution (Legitimate vs Fraudulent)</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Visualizing the severe transaction imbalance: 284,315 legitimate records vs 492 fraud cases.</div>", unsafe_allow_html=True)
        df_dist = pd.DataFrame({
            "Classification": ["Legitimate (99.828%)", "Fraudulent (0.172%)"],
            "Transactions": [284315, 492]
        })
        fig_donut = px.pie(
            df_dist, names="Classification", values="Transactions",
            color="Classification",
            color_discrete_map={"Legitimate (99.828%)": "#2563EB", "Fraudulent (0.172%)": "#EF4444"},
            hole=0.55
        )
        apply_fintech_chart_theme(fig_donut, height=240)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("""
        <div class='chart-explanation'>
            <b>Understanding the Class Skew:</b> Fraud accounts for only 0.172% of all historical records in this dataset. A naive classifier that predicts "Legitimate" for every record would achieve 99.83% accuracy while capturing 0% of fraud. Model evaluation must therefore rely on Precision, Recall, and PR-AUC.
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visualizations: Row 2 - 48-Hour Activity & Fraud Occurrence Over Time
    c_act1, c_act2 = st.columns(2)
    with c_act1:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Transaction Activity Over Time (48-Hour Cycle)</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Total hourly payment traffic across the 2-day historical observation window.</div>", unsafe_allow_html=True)
            if not hourly_df.empty:
                fig_vol = px.bar(
                    hourly_df, x="Hour", y="Total_Transactions",
                    color_discrete_sequence=["#3B82F6"]
                )
                apply_fintech_chart_theme(fig_vol, height=260)
                fig_vol.update_xaxes(title_text="Timeline (Hour 0 to Hour 47)")
                fig_vol.update_yaxes(title_text="Transaction Count")
                st.plotly_chart(fig_vol, use_container_width=True)
                st.markdown("""
                <div class='chart-explanation'>
                    <b>Activity Pattern:</b> The Time feature records elapsed seconds from the first recorded transaction. Aggregated hourly volume exhibits periodic fluctuations over the 48-hour window, with peak intervals exceeding 8,000 transactions per hour and recurring troughs around hours 1–6 and 25–30.
                </div>
                """, unsafe_allow_html=True)

    with c_act2:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Fraud Occurrence Over Time (48-Hour Cycle)</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Hourly distribution of confirmed fraudulent events across the 48-hour period.</div>", unsafe_allow_html=True)
            if not hourly_df.empty:
                fig_fraud_bar = px.bar(
                    hourly_df, x="Hour", y="Fraud_Events",
                    color="Fraud_Events",
                    color_continuous_scale=["#FECACA", "#EF4444"]
                )
                apply_fintech_chart_theme(fig_fraud_bar, height=260)
                fig_fraud_bar.update_layout(coloraxis_showscale=False)
                fig_fraud_bar.update_xaxes(title_text="Timeline (Hour 0 to Hour 47)")
                fig_fraud_bar.update_yaxes(title_text="Confirmed Fraud Events")
                st.plotly_chart(fig_fraud_bar, use_container_width=True)
                st.markdown("""
                <div class='chart-explanation'>
                    <b>Observed Distribution:</b> Fraudulent transactions are distributed across the recorded timeline, with the highest counts occurring at Hour 11 (43 events) and Hour 26 (36 events). Fraud events continue across both high-volume and low-volume intervals.
                </div>
                """, unsafe_allow_html=True)


# =====================================================================
# PAGE 2: DETECT FRAUD - THE MAIN FEATURE
# =====================================================================
elif menu_selection == "Detect Fraud":
    st.markdown("""
    <div class='portal-topbar'>
        <div class='portal-title-box'>
            <h2 class='portal-title'>Detect Fraud</h2>
            <p class='portal-sub'>Test individual transactions using trained Machine Learning & Deep Learning models</p>
        </div>
        <div class='portal-badge'>
            ● SINGLE TRANSACTION EVALUATION
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Session state initialization for single transaction testing
    if "base_record_id" not in st.session_state:
        st.session_state["base_record_id"] = "Sample Legitimate Record"
    if "screen_amount_eur" not in st.session_state:
        st.session_state["screen_amount_eur"] = 110.28
    if "base_amount_eur" not in st.session_state:
        st.session_state["base_amount_eur"] = 110.28
    if "screen_time" not in st.session_state:
        st.session_state["screen_time"] = 125704.0
    if "base_time_sec" not in st.session_state:
        st.session_state["base_time_sec"] = 125704.0
    if "ground_truth_class" not in st.session_state:
        st.session_state["ground_truth_class"] = 0
    if "has_evaluated" not in st.session_state:
        st.session_state["has_evaluated"] = False
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None
    if "consensus_result" not in st.session_state:
        st.session_state["consensus_result"] = None
    for i in range(1, 29):
        if f"screen_v_{i}" not in st.session_state:
            st.session_state[f"screen_v_{i}"] = 0.0

    def load_sample_record(row: pd.Series, desc: str):
        st.session_state["base_record_id"] = desc
        amt_eur = float(row["Amount"])
        time_sec = float(row["Time"])
        st.session_state["screen_amount_eur"] = amt_eur
        st.session_state["base_amount_eur"] = amt_eur
        st.session_state["screen_time"] = time_sec
        st.session_state["base_time_sec"] = time_sec
        st.session_state["ground_truth_class"] = int(row["Class"])
        st.session_state["has_evaluated"] = False
        st.session_state["eval_result"] = None
        st.session_state["consensus_result"] = None
        for i in range(1, 29):
            st.session_state[f"screen_v_{i}"] = float(row.get(f"V{i}", 0.0))

    # STEP 1: LOAD A SAMPLE TRANSACTION (3 SIMPLE OPTIONS)
    with st.container(border=True):
        st.markdown("<div class='section-title'>Step 1: Load a Sample Transaction</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Select a transaction from the test pool to populate the feature parameters:</div>", unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("Sample Legitimate Record (€110.28)", use_container_width=True, type="secondary"):
                if not test_pool_df.empty:
                    legit_rows = test_pool_df[test_pool_df["Class"] == 0]
                    chosen = legit_rows.iloc[0]
                    load_sample_record(chosen, f"Dataset Record #{int(chosen.get('Dataset_Row_Index', 183218))} (Legitimate)")
                    st.rerun()

        with sc2:
            if st.button("Sample Fraudulent Record (€88.23)", use_container_width=True, type="secondary"):
                if not test_pool_df.empty:
                    fraud_rows = test_pool_df[test_pool_df["Class"] == 1]
                    chosen = fraud_rows.iloc[1] if len(fraud_rows) > 1 else fraud_rows.iloc[0]
                    load_sample_record(chosen, f"Dataset Record #{int(chosen.get('Dataset_Row_Index', 4920))} (Fraudulent)")
                    st.rerun()

        with sc3:
            if st.button("Random Test Record", use_container_width=True, type="secondary"):
                if not test_pool_df.empty:
                    chosen = test_pool_df.sample(n=1).iloc[0]
                    label_desc = "Fraudulent" if int(chosen.get("Class", 0)) == 1 else "Legitimate"
                    load_sample_record(chosen, f"Dataset Record #{int(chosen.get('Dataset_Row_Index', 0))} ({label_desc})")
                    st.rerun()

    st.write("")

    # STEP 2: TRANSACTION PARAMETERS & DETAILS
    with st.container(border=True):
        st.markdown("<div class='section-title'>Step 2: Transaction Parameters</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Inspect transaction amount and elapsed recording time:</div>", unsafe_allow_html=True)

        # Reference Banner
        label_text = "Confirmed Fraud (Class 1)" if st.session_state["ground_truth_class"] == 1 else "Legitimate (Class 0)"
        st.markdown(f"""
        <div style='background:#F1F5F9; border:1px solid #CBD5E1; border-radius:6px; padding:8px 14px; margin-bottom:14px; font-size:0.84rem; color:#334155;'>
            <b>Dataset Reference:</b> {st.session_state.get('base_record_id', 'Sample Record')} &nbsp;•&nbsp; <b>Ground Truth Label:</b> {label_text}
        </div>
        """, unsafe_allow_html=True)

        col_amt_in, col_timing_in = st.columns(2)
        with col_amt_in:
            if is_inr:
                curr_inr = float(round(st.session_state["screen_amount_eur"] * config.EUR_TO_INR_RATE, 2))
                entered_val = st.number_input(
                    "Transaction Amount (₹ INR):",
                    min_value=0.10,
                    max_value=2500000.0,
                    value=curr_inr,
                    step=50.0,
                    format="%.2f",
                    help="Amount converted to original EUR for model inference."
                )
                st.session_state["screen_amount_eur"] = entered_val / config.EUR_TO_INR_RATE
            else:
                curr_eur = float(round(st.session_state["screen_amount_eur"], 2))
                entered_val = st.number_input(
                    "Transaction Amount (€ EUR):",
                    min_value=0.01,
                    max_value=25000.0,
                    value=curr_eur,
                    step=5.0,
                    format="%.2f",
                    help="Transaction amount in original EUR."
                )
                st.session_state["screen_amount_eur"] = entered_val

        with col_timing_in:
            time_val = float(st.session_state["screen_time"])
            entered_time = st.number_input(
                "Elapsed Time Since Start (seconds):",
                min_value=0.0,
                max_value=172800.0,
                value=time_val,
                step=100.0,
                format="%.1f",
                help="Time elapsed in seconds since the first recorded transaction (0 to 172,800s)."
            )
            st.session_state["screen_time"] = entered_time
            st.caption(f"Equivalent to Hour {entered_time / 3600.0:.2f} within the 48-hour recording timeline.")

        # Check for Modified Test Scenario
        amt_diff = abs(st.session_state["screen_amount_eur"] - st.session_state.get("base_amount_eur", st.session_state["screen_amount_eur"]))
        time_diff = abs(st.session_state["screen_time"] - st.session_state.get("base_time_sec", st.session_state["screen_time"]))
        if amt_diff > 0.01 or time_diff > 1.0:
            st.markdown(f"""
            <div style='background-color:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid #F59E0B; padding:10px 14px; border-radius:6px; margin-top:10px; color:#92400E; font-size:0.84rem;'>
                <b>Modified Test Scenario:</b> Transaction amount or elapsed time has been customized while retaining the underlying PCA components from {st.session_state.get('base_record_id', 'the selected record')}. This represents a synthetic test scenario, not an original historical dataset record.
            </div>
            """, unsafe_allow_html=True)

        # Collapsible PCA components
        st.write("")
        with st.expander("Optional: View Technical PCA Features", expanded=False):
            st.caption(
                "Features V1 through V28 are anonymized numerical principal components produced via PCA on the original European dataset. "
                "These 28 numerical features are passed to the model alongside Amount and Time."
            )
            v_preview = {f"V{i}": [round(st.session_state[f"screen_v_{i}"], 4)] for i in range(1, 29)}
            st.dataframe(pd.DataFrame(v_preview), use_container_width=True, hide_index=True)

    st.write("")

    # STEP 3: MODEL SELECTION & INFERENCE
    with st.container(border=True):
        st.markdown("<div class='section-title'>Step 3: Select Model Architecture & Run Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Select a trained model to evaluate the 30-feature transaction vector:</div>", unsafe_allow_html=True)

        col_m, col_adv = st.columns([3, 2])
        with col_m:
            chosen_model = st.selectbox(
                "Select Model:",
                [
                    "Random Forest",
                    "Artificial Neural Network",
                    "Logistic Regression",
                    "Autoencoder"
                ]
            )

        default_thresholds = {
            "Random Forest": 0.8456,
            "Artificial Neural Network": 0.9000,
            "Logistic Regression": 0.9000,
            "Autoencoder": 3.0607
        }
        thresh_val = default_thresholds.get(chosen_model, 0.9000)

        with col_adv:
            enable_advanced_tuning = st.checkbox("Adjust Decision Threshold", value=False)
            if enable_advanced_tuning:
                if chosen_model == "Autoencoder":
                    thresh_val = st.slider(
                        "Anomaly Cutoff Threshold (MSE)",
                        min_value=1.0, max_value=6.0, value=3.0607, step=0.05,
                        help="Reconstruction MSE exceeding this threshold flags the record as an anomaly."
                    )
                else:
                    thresh_val = st.slider(
                        "Decision Threshold",
                        min_value=0.10, max_value=0.99, value=thresh_val, step=0.01,
                        help="Probability cutoff for fraud classification."
                    )
            else:
                st.caption(f"Decision Threshold: `{thresh_val}` (Empirically calibrated benchmark).")

        st.write("")
        btn_run = st.button("ANALYSE TRANSACTION", type="primary", use_container_width=True)

        if btn_run:
            payload = {
                "Amount": float(st.session_state["screen_amount_eur"]),
                "Time": float(st.session_state["screen_time"])
            }
            for i in range(1, 29):
                payload[f"V{i}"] = float(st.session_state.get(f"screen_v_{i}", 0.0))

            with st.spinner("Evaluating transaction features..."):
                res = engine.predict_single_transaction(payload, model_name=chosen_model, threshold=thresh_val)
                consensus = engine.compare_all_models(payload)

            st.session_state["has_evaluated"] = True
            st.session_state["eval_result"] = res
            st.session_state["consensus_result"] = consensus

    # STEP 4: PREDICTION RESULT
    if st.session_state.get("has_evaluated", False) and st.session_state["eval_result"] is not None:
        res = st.session_state["eval_result"]
        consensus = st.session_state["consensus_result"]
        is_fraud = res["is_fraud"]

        if chosen_model == "Autoencoder":
            raw_mse = res["anomaly_score"] or 0.0
            score_label = f"{raw_mse:.4f} MSE"
            score_type_label = "Anomaly Score (Reconstruction MSE)"
        else:
            prob = res["fraud_probability"] or 0.0
            score_label = f"{prob * 100:.2f}%"
            score_type_label = "Model Probability Score"

        st.write("")
        with st.container(border=True):
            st.markdown("<div class='section-title'>Step 4: Analysis Result</div>", unsafe_allow_html=True)

            if is_fraud:
                st.markdown(f"""
                <div style='background-color:#FEF2F2; border:2px solid #EF4444; border-radius:10px; padding:18px 20px; margin-bottom:14px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:#991B1B; font-size:1.25rem; font-weight:800;'>PREDICTION: POTENTIALLY FRAUDULENT</span>
                        <span style='background:#FEE2E2; border:1px solid #FCA5A5; color:#991B1B; font-weight:700; font-size:0.78rem; padding:4px 10px; border-radius:6px;'>Class 1 Flagged</span>
                    </div>
                    <div style='margin-top:10px; font-size:0.90rem; color:#1E293B; line-height:1.6;'>
                        <b>Model Applied:</b> {chosen_model}<br>
                        <b>{score_type_label}:</b> <span style='color:#DC2626; font-weight:800;'>{score_label}</span> (Decision Threshold: {thresh_val})<br>
                        <b>Technical Assessment:</b> The input feature vector exhibits mathematical deviations characteristic of confirmed fraudulent transactions in the training data.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#F0FDF4; border:2px solid #22C55E; border-radius:10px; padding:18px 20px; margin-bottom:14px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:#166534; font-size:1.25rem; font-weight:800;'>PREDICTION: LEGITIMATE</span>
                        <span style='background:#DCFCE7; border:1px solid #86EFAC; color:#166534; font-weight:700; font-size:0.78rem; padding:4px 10px; border-radius:6px;'>Class 0 Classified</span>
                    </div>
                    <div style='margin-top:10px; font-size:0.90rem; color:#1E293B; line-height:1.6;'>
                        <b>Model Applied:</b> {chosen_model}<br>
                        <b>{score_type_label}:</b> <span style='color:#16A34A; font-weight:800;'>{score_label}</span> (Decision Threshold: {thresh_val})<br>
                        <b>Technical Assessment:</b> The input feature vector aligns with the learned multivariate distribution of authentic non-fraudulent cardholder records.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Consensus across all 4 models
            st.write("")
            st.markdown("<div class='section-title'>Multi-Model Evaluation Summary</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Comparison across all 4 trained architectures for this transaction:</div>", unsafe_allow_html=True)

            indiv = consensus["individual_results"] if consensus else {}
            cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

            with cm_col1:
                rf_data = indiv.get("Random Forest", {})
                rf_prob = rf_data.get("fraud_probability", 0.0) or 0.0
                rf_flag = rf_data.get("is_fraud", False)
                rf_status = "Potentially Fraudulent" if rf_flag else "Legitimate"
                rf_status_col = "#DC2626" if rf_flag else "#16A34A"
                st.markdown(f"""
                <div class='consensus-metric-card'>
                    <div class='consensus-metric-title'>Random Forest</div>
                    <div class='consensus-metric-val'>{rf_prob*100:.1f}%</div>
                    <span style='color:{rf_status_col}; font-weight:700; font-size:0.80rem;'>{rf_status}</span>
                </div>
                """, unsafe_allow_html=True)

            with cm_col2:
                ann_data = indiv.get("Artificial Neural Network", {})
                ann_prob = ann_data.get("fraud_probability", 0.0) or 0.0
                ann_flag = ann_data.get("is_fraud", False)
                ann_status = "Potentially Fraudulent" if ann_flag else "Legitimate"
                ann_status_col = "#DC2626" if ann_flag else "#16A34A"
                st.markdown(f"""
                <div class='consensus-metric-card'>
                    <div class='consensus-metric-title'>Neural Network (ANN)</div>
                    <div class='consensus-metric-val'>{ann_prob*100:.1f}%</div>
                    <span style='color:{ann_status_col}; font-weight:700; font-size:0.80rem;'>{ann_status}</span>
                </div>
                """, unsafe_allow_html=True)

            with cm_col3:
                lr_data = indiv.get("Logistic Regression", {})
                lr_prob = lr_data.get("fraud_probability", 0.0) or 0.0
                lr_flag = lr_data.get("is_fraud", False)
                lr_status = "Potentially Fraudulent" if lr_flag else "Legitimate"
                lr_status_col = "#DC2626" if lr_flag else "#16A34A"
                st.markdown(f"""
                <div class='consensus-metric-card'>
                    <div class='consensus-metric-title'>Logistic Regression</div>
                    <div class='consensus-metric-val'>{lr_prob*100:.1f}%</div>
                    <span style='color:{lr_status_col}; font-weight:700; font-size:0.80rem;'>{lr_status}</span>
                </div>
                """, unsafe_allow_html=True)

            with cm_col4:
                ae_data = indiv.get("Autoencoder", {})
                ae_mse = ae_data.get("anomaly_score", 0.0) or 0.0
                ae_flag = ae_data.get("is_fraud", False)
                ae_status = "Anomaly Flagged" if ae_flag else "Normal Pattern"
                ae_status_col = "#DC2626" if ae_flag else "#16A34A"
                st.markdown(f"""
                <div class='consensus-metric-card'>
                    <div class='consensus-metric-title'>Autoencoder (Anomaly)</div>
                    <div class='consensus-metric-val'>{ae_mse:.3f} MSE</div>
                    <span style='color:{ae_status_col}; font-weight:700; font-size:0.80rem;'>{ae_status}</span>
                </div>
                """, unsafe_allow_html=True)

            # Ground Truth Audit
            st.write("")
            actual_label = st.session_state["ground_truth_class"]
            label_str = "Confirmed Fraud (Class = 1)" if actual_label == 1 else "Legitimate (Class = 0)"
            match_str = "Prediction aligns with historical dataset ground truth" if (is_fraud == (actual_label == 1)) else "Prediction diverges from historical dataset label"
            match_color = "#166534" if (is_fraud == (actual_label == 1)) else "#991B1B"
            st.markdown(f"""
            <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 14px; font-size:0.84rem; color:#475569;'>
                <b>Dataset Ground Truth Audit:</b> <b>{label_str}</b> &nbsp;•&nbsp; <span style='color:{match_color}; font-weight:700;'>{match_str}</span><br>
                <span style='font-size:0.78rem; color:#64748B;'>Note: The dataset Class label was strictly excluded from all model inputs during inference.</span>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# PAGE 3: BATCH DETECTION
# =====================================================================
elif menu_selection == "Batch Detection":
    st.markdown("""
    <div class='portal-topbar'>
        <div class='portal-title-box'>
            <h2 class='portal-title'>Batch Detection</h2>
            <p class='portal-sub'>Run automated fraud screening on CSV files containing multiple payment records</p>
        </div>
        <div class='portal-badge'>
            ● HIGH-THROUGHPUT AUDITING
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "batch_input_df" not in st.session_state:
        st.session_state["batch_input_df"] = None
    if "batch_output_results" not in st.session_state:
        st.session_state["batch_output_results"] = None

    # Step 1: Requirements & Template
    with st.container(border=True):
        st.markdown("<div class='section-title'>File Requirements & Template</div>", unsafe_allow_html=True)
        st.markdown("""
        Upload a standard UTF-8 **CSV file** containing transaction parameters for automated risk scoring.
        - **Required Columns (30 input features):** `Time`, `V1` to `V28` (numerical PCA components), and `Amount` (Euros).
        - **Optional Column:** `Class` (historical label, if present, is ignored during model prediction).
        - **Validation:** All values must be numeric; missing or null values will be flagged.
        """)

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if not batch_template_df.empty:
                csv_bytes = batch_template_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Sample CSV Template (50 Transactions)",
                    data=csv_bytes,
                    file_name="finsecure_sample_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with b_c2:
            if st.button("Quick Test: Load Sample Batch (50 Rows)", use_container_width=True, type="secondary"):
                if not batch_template_df.empty:
                    st.session_state["batch_input_df"] = batch_template_df.copy()
                    st.session_state["batch_output_results"] = None
                    st.success("Loaded 50 verified sample transactions containing a mix of legitimate and fraudulent test cases.")

    st.write("")

    # Step 2: Upload CSV
    with st.container(border=True):
        st.markdown("<div class='section-title'>Upload Transaction CSV</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                if raw_df.empty:
                    st.error("Uploaded CSV file is completely empty.")
                else:
                    needed = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN]
                    missing = [c for c in needed if c not in raw_df.columns]
                    if missing:
                        st.error(f"Validation Error: File is missing required columns: {missing[:5]} (Total missing: {len(missing)})")
                    else:
                        nulls = int(raw_df[needed].isnull().sum().sum())
                        if nulls > 0:
                            st.error(f"Validation Error: Dataset contains {nulls} missing/null values.")
                        else:
                            st.session_state["batch_input_df"] = raw_df
                            st.session_state["batch_output_results"] = None
                            st.success(f"File validated successfully: {len(raw_df):,} transactions ready for screening.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        # Transaction Preview
        if st.session_state["batch_input_df"] is not None:
            df_curr = st.session_state["batch_input_df"]
            with st.expander(f"Transaction Preview ({len(df_curr):,} Total Rows - Showing First 5)", expanded=False):
                disp_preview_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 6)]
                st.dataframe(df_curr[disp_preview_cols].head(5), use_container_width=True, hide_index=True)

            st.write("")
            m_col, btn_col, rst_col = st.columns([3, 1, 1])
            with m_col:
                batch_model = st.selectbox(
                    "Select Auditing Model:",
                    ["Random Forest", "Artificial Neural Network", "Logistic Regression", "Autoencoder"]
                )
            with btn_col:
                st.write("")
                st.write("")
                run_batch = st.button("RUN BATCH DETECTION", type="primary", use_container_width=True)
            with rst_col:
                st.write("")
                st.write("")
                if st.button("Reset Batch", use_container_width=True, type="secondary"):
                    st.session_state["batch_input_df"] = None
                    st.session_state["batch_output_results"] = None
                    st.rerun()

            if run_batch:
                with st.spinner(f"Screening {len(df_curr):,} transactions using {batch_model}..."):
                    st.session_state["batch_output_results"] = engine.predict_batch(df_curr, model_name=batch_model)

            # Display Batch Results
            if st.session_state["batch_output_results"] is not None:
                b_res = st.session_state["batch_output_results"]
                res_df = b_res["results_dataframe"]

                st.write("")
                st.markdown("<div class='section-title'>Batch Screening Summary</div>", unsafe_allow_html=True)
                bk1, bk2, bk3, bk4 = st.columns(4)
                with bk1:
                    st.metric("Total Processed", f"{b_res['total_transactions']:,}")
                with bk2:
                    st.metric("Legitimate Predictions", f"{b_res['legitimate_count']:,}")
                with bk3:
                    st.metric("Potentially Fraudulent", f"{b_res['fraud_count']:,}", delta=f"{b_res['fraud_rate_percentage']:.2f}% Flagged", delta_color="inverse")
                with bk4:
                    st.metric("Model Applied", b_res["model_used"])

                st.write("")
                # Filter results
                filt_choice = st.radio("Filter Table View:", ["All Transactions", "Only Potentially Fraudulent", "Only Legitimate"], horizontal=True)
                view_df = res_df.copy()
                if filt_choice == "Only Potentially Fraudulent":
                    view_df = view_df[view_df["Is_Fraud"] == 1]
                elif filt_choice == "Only Legitimate":
                    view_df = view_df[view_df["Is_Fraud"] == 0]

                # Format table columns for clear, accurate display
                is_ae = (b_res["model_used"] == "Autoencoder")
                score_display_col = "Anomaly Score (MSE)" if is_ae else "Fraud Score / Probability"
                score_raw_col = "Reconstruction_MSE" if is_ae else "Fraud_Probability"

                view_df["Recorded Time (seconds)"] = view_df["Time"].apply(lambda t: f"{t:,.1f}")
                view_df["Transaction Amount"] = view_df["Amount"].apply(fmt_money_short)
                view_df["Risk Level"] = view_df["Risk_Level"]

                if is_ae:
                    view_df[score_display_col] = view_df[score_raw_col].apply(lambda s: f"{float(s):.4f} MSE")
                else:
                    view_df[score_display_col] = view_df[score_raw_col].apply(lambda s: f"{float(s):.5f}")

                show_cols = ["Recorded Time (seconds)", "Transaction Amount", "Prediction", "Risk Level", score_display_col]
                st.dataframe(view_df[show_cols].head(100), use_container_width=True, hide_index=True)

                # Download Results CSV
                scored_csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Scored Results as CSV",
                    data=scored_csv,
                    file_name=f"finsecure_scored_results_{b_res['model_used'].lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    type="primary"
                )


# =====================================================================
# PAGE 4: EDA & MODEL PERFORMANCE (2 CLEAR TABS)
# =====================================================================
elif menu_selection == "EDA & Model Performance":
    st.markdown("""
    <div class='portal-topbar'>
        <div class='portal-title-box'>
            <h2 class='portal-title'>EDA & Model Performance</h2>
            <p class='portal-sub'>Empirical data distributions and verified benchmark evaluations across all 4 machine learning and deep learning architectures</p>
        </div>
        <div class='portal-badge'>
            ● EMPIRICAL BENCHMARKS
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_eda, tab_perf = st.tabs(["Exploratory Data Analysis (EDA)", "Model Performance & Benchmarks"])

    # -------------------------------------------------------------
    # TAB 1: EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    with tab_eda:
        st.markdown("<div class='section-title'>Key Empirical Distributions</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Statistical distributions and correlation patterns derived from the 284,807 European cardholder transactions.</div>", unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            with st.container(border=True):
                st.markdown("<div class='section-title'>1. Class Distribution (Extreme Skew)</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>284,315 clean transactions (99.828%) vs 492 fraud cases (0.172%).</div>", unsafe_allow_html=True)
                fig_c = px.pie(
                    values=[284315, 492],
                    names=["Legitimate (99.828%)", "Fraudulent (0.172%)"],
                    color_discrete_sequence=["#2563EB", "#EF4444"],
                    hole=0.5
                )
                apply_fintech_chart_theme(fig_c, height=240)
                st.plotly_chart(fig_c, use_container_width=True)
                st.markdown("""
                <div class='chart-explanation'>
                    <b>Explanation:</b> Only 1 out of ~580 transactions is fraudulent. Because class 0 overwhelmingly dominates, standard accuracy is an invalid metric. Models must be evaluated on Precision, Recall, and PR-AUC.
                </div>
                """, unsafe_allow_html=True)

        with e2:
            with st.container(border=True):
                st.markdown("<div class='section-title'>2. Transaction Amount Distribution</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>Distribution of purchase charges across value brackets.</div>", unsafe_allow_html=True)
                amt_b = pd.DataFrame({
                    "Bracket": ["Under €10", "€10 – €50", "€50 – €100", "€100 – €500", "Over €500"],
                    "Transactions": [101485, 96328, 42116, 38104, 6774]
                })
                fig_ab = px.bar(
                    amt_b, x="Bracket", y="Transactions",
                    color_discrete_sequence=["#3B82F6"]
                )
                apply_fintech_chart_theme(fig_ab, height=240)
                st.plotly_chart(fig_ab, use_container_width=True)
                st.markdown("""
                <div class='chart-explanation'>
                    <b>Explanation:</b> Over 69% of all card purchases are under €50, with a median of €22.00. The distribution exhibits extreme positive skewness, requiring RobustScaler normalization based on median and IQR.
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        e3, e4 = st.columns(2)
        with e3:
            with st.container(border=True):
                st.markdown("<div class='section-title'>3. Legitimate vs Fraud Amount Comparison (Log Scale)</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>Comparing transaction amounts between clean and fraudulent events.</div>", unsafe_allow_html=True)
                if not sample_df.empty:
                    legit_amts = sample_df[sample_df["Class"] == 0]["Amount"]
                    fraud_amts = sample_df[sample_df["Class"] == 1]["Amount"]
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=legit_amts, name="Legitimate", marker_color="#2563EB", boxpoints=False))
                    fig_box.add_trace(go.Box(y=fraud_amts, name="Fraudulent", marker_color="#EF4444", boxpoints=False))
                    apply_fintech_chart_theme(fig_box, height=260)
                    fig_box.update_yaxes(type="log", title_text="Amount (€ - Log Scale)")
                    st.plotly_chart(fig_box, use_container_width=True)
                    st.markdown("""
                    <div class='chart-explanation'>
                        <b>Explanation:</b> In this historical dataset, recorded legitimate transactions have a median of €22.00 (mean €88.29), while recorded fraudulent transactions have a lower median of €9.25 (mean €122.21). Although fraudulent amounts extend up to €2,125.87, over half of all confirmed fraudulent transactions are under €10.00.
                    </div>
                    """, unsafe_allow_html=True)

        with e4:
            with st.container(border=True):
                st.markdown("<div class='section-title'>4. Transaction Activity Over Time (48-Hour Cycle)</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>Hourly transaction traffic across the 2-day historical observation timeline.</div>", unsafe_allow_html=True)
                if not hourly_df.empty:
                    fig_tl = px.line(
                        hourly_df, x="Hour", y="Total_Transactions",
                        markers=True, line_shape="spline"
                    )
                    fig_tl.update_traces(line_color="#2563EB")
                    apply_fintech_chart_theme(fig_tl, height=260)
                    fig_tl.update_xaxes(title_text="Timeline (Hour 0 to Hour 47)")
                    fig_tl.update_yaxes(title_text="Transaction Count")
                    st.plotly_chart(fig_tl, use_container_width=True)
                    st.markdown("""
                    <div class='chart-explanation'>
                        <b>Explanation:</b> The Time column represents elapsed seconds from the first recorded transaction across the 48-hour recording timeline. Aggregated volume demonstrates periodic cyclical fluctuations with notable low-volume intervals around 10,000–25,000 seconds and 95,000–110,000 seconds.
                    </div>
                    """, unsafe_allow_html=True)

        st.write("")

        e5, e6 = st.columns(2)
        with e5:
            with st.container(border=True):
                st.markdown("<div class='section-title'>5. Fraud Occurrence Over Time</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>Hourly frequency of confirmed fraudulent transactions across the 48-hour recording.</div>", unsafe_allow_html=True)
                if not hourly_df.empty:
                    fig_fb = px.bar(
                        hourly_df, x="Hour", y="Fraud_Events",
                        color="Fraud_Events", color_continuous_scale=["#FECACA", "#EF4444"]
                    )
                    apply_fintech_chart_theme(fig_fb, height=260)
                    fig_fb.update_layout(coloraxis_showscale=False)
                    fig_fb.update_xaxes(title_text="Timeline (Hour 0 to Hour 47)")
                    fig_fb.update_yaxes(title_text="Confirmed Fraud Events")
                    st.plotly_chart(fig_fb, use_container_width=True)
                    st.markdown("""
                    <div class='chart-explanation'>
                        <b>Explanation:</b> Fraudulent occurrences are distributed across the recorded 48-hour window, with observed peaks at Hour 11 (43 events) and Hour 26 (36 events). Confirmed fraud events are present across both high-volume and low-volume intervals.
                    </div>
                    """, unsafe_allow_html=True)

        with e6:
            with st.container(border=True):
                st.markdown("<div class='section-title'>6. Feature Correlations with Fraud (PCA Components)</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-subtitle'>Pearson correlation coefficients (r) of anonymized PCA components with fraud.</div>", unsafe_allow_html=True)
                df_corr = pd.DataFrame([
                    {"Feature": "V17", "Correlation": -0.326},
                    {"Feature": "V14", "Correlation": -0.302},
                    {"Feature": "V12", "Correlation": -0.260},
                    {"Feature": "V10", "Correlation": -0.217},
                    {"Feature": "V11", "Correlation": 0.155},
                    {"Feature": "V4",  "Correlation": 0.133}
                ])
                fig_cr = px.bar(
                    df_corr, x="Correlation", y="Feature", orientation="h",
                    color="Correlation", color_continuous_scale=["#EF4444", "#CBD5E1", "#2563EB"]
                )
                apply_fintech_chart_theme(fig_cr, height=260)
                fig_cr.update_layout(coloraxis_showscale=False)
                fig_cr.update_xaxes(title_text="Correlation Coefficient (r)")
                st.plotly_chart(fig_cr, use_container_width=True)
                st.markdown("""
                <div class='chart-explanation'>
                    <b>Explanation:</b> Anonymized components V17 (r = -0.326), V14 (r = -0.302), and V12 (r = -0.260) show the highest negative linear correlation with Class in this dataset, while V11 (r = 0.155) and V4 (r = 0.133) show moderate positive correlation. These metrics indicate statistical associations, not causal relationships.
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        # Key Findings
        with st.container(border=True):
            st.markdown("<div class='section-title'>Key Analytical Findings</div>", unsafe_allow_html=True)
            st.markdown("""
            1. **Severe Imbalance:** Fraud represents 0.172% of total dataset records (492 confirmed fraud cases among 284,807 total transactions).
            2. **Transaction Value Distribution:** The median transaction amount is €22.00 for legitimate transactions and €9.25 for fraudulent transactions; both distributions are heavily right-skewed.
            3. **Elapsed Time Distribution:** Aggregated transaction volume demonstrates periodic cyclical waves across the 48-hour recording timeline, while fraudulent transactions are recorded across all intervals.
            4. **Feature Association:** PCA components V17, V14, and V12 exhibit the strongest negative linear correlation with the target Class label.
            """)

    # -------------------------------------------------------------
    # TAB 2: MODEL PERFORMANCE & BENCHMARKS
    # -------------------------------------------------------------
    with tab_perf:
        st.markdown("<div class='section-title'>Model Performance Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Empirically measured across 42,722 held-out test transactions (74 confirmed fraud cases, 42,648 legitimate cases).</div>", unsafe_allow_html=True)

        # The Accuracy Paradox Callout
        st.markdown("""
        <div style='background-color:#EFF6FF; border-left:4px solid #2563EB; padding:14px 18px; border-radius:8px; margin-bottom:16px;'>
            <b style='color:#1E3A8A; font-size:0.95rem;'>The Accuracy Paradox in Fraud Detection:</b><br>
            <span style='color:#1E293B; font-size:0.88rem; line-height:1.5;'>
            In an imbalanced dataset where 99.828% of transactions are legitimate, a useless dummy model that predicts <i>everything is legitimate</i> achieves <b>99.83% accuracy</b> while capturing <b>0% of fraud attacks</b>.
            Therefore, real-world fraud models must prioritize <b>Precision</b> (few false declines), <b>Recall</b> (% of fraud intercepted), and <b>PR-AUC</b> over raw accuracy.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Master Comparison Table & Visual Performance Chart
        with st.container(border=True):
            st.markdown("<div class='section-title'>Empirical Model Benchmark Comparison (42,722 Test Samples)</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Comparing Precision, Recall, and F1-Score across all 4 machine learning and deep learning architectures:</div>", unsafe_allow_html=True)

            # Visual Grouped Bar Chart
            perf_chart_df = pd.DataFrame([
                {"Model": "Random Forest", "Metric": "Precision", "Score (%)": 91.53},
                {"Model": "Random Forest", "Metric": "Recall", "Score (%)": 72.97},
                {"Model": "Random Forest", "Metric": "F1-Score", "Score (%)": 81.20},
                {"Model": "Neural Network (ANN)", "Metric": "Precision", "Score (%)": 57.41},
                {"Model": "Neural Network (ANN)", "Metric": "Recall", "Score (%)": 83.78},
                {"Model": "Neural Network (ANN)", "Metric": "F1-Score", "Score (%)": 68.13},
                {"Model": "Logistic Regression", "Metric": "Precision", "Score (%)": 23.42},
                {"Model": "Logistic Regression", "Metric": "Recall", "Score (%)": 85.14},
                {"Model": "Logistic Regression", "Metric": "F1-Score", "Score (%)": 36.73},
                {"Model": "Autoencoder", "Metric": "Precision", "Score (%)": 40.31},
                {"Model": "Autoencoder", "Metric": "Recall", "Score (%)": 70.27},
                {"Model": "Autoencoder", "Metric": "F1-Score", "Score (%)": 51.23},
            ])
            fig_perf = px.bar(
                perf_chart_df, x="Model", y="Score (%)", color="Metric",
                barmode="group",
                color_discrete_map={"Precision": "#2563EB", "Recall": "#10B981", "F1-Score": "#8B5CF6"}
            )
            apply_fintech_chart_theme(fig_perf, height=270)
            fig_perf.update_yaxes(title_text="Percentage (%)", range=[0, 100])
            st.plotly_chart(fig_perf, use_container_width=True)

            st.write("")
            table_data = [
                {
                    "Model": "Random Forest",
                    "Type": "Ensemble Supervised ML",
                    "Accuracy": "99.94%",
                    "Precision": "91.53%",
                    "Recall": "72.97% (54/74)",
                    "F1-Score": "0.8120",
                    "ROC-AUC": "0.9756",
                    "PR-AUC": "0.7953",
                    "False Positives": "5 (0.012%)",
                    "Threshold": "0.8456"
                },
                {
                    "Model": "Artificial Neural Network",
                    "Type": "Supervised Deep Learning",
                    "Accuracy": "99.86%",
                    "Precision": "57.41%",
                    "Recall": "83.78% (62/74)",
                    "F1-Score": "0.6813",
                    "ROC-AUC": "0.9748",
                    "PR-AUC": "0.7745",
                    "False Positives": "46 (0.108%)",
                    "Threshold": "0.9000"
                },
                {
                    "Model": "Logistic Regression",
                    "Type": "Linear Supervised ML",
                    "Accuracy": "99.49%",
                    "Precision": "23.42%",
                    "Recall": "85.14% (63/74)",
                    "F1-Score": "0.3673",
                    "ROC-AUC": "0.9641",
                    "PR-AUC": "0.7821",
                    "False Positives": "206 (0.483%)",
                    "Threshold": "0.9000"
                },
                {
                    "Model": "Autoencoder",
                    "Type": "Unsupervised Deep Learning",
                    "Accuracy": "99.77%",
                    "Precision": "40.31%",
                    "Recall": "70.27% (52/74)",
                    "F1-Score": "0.5123",
                    "ROC-AUC": "0.9144",
                    "PR-AUC": "0.4982",
                    "False Positives": "77 (0.181%)",
                    "Threshold": "3.0607 MSE"
                }
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.write("")

        # Confusion Matrices Breakdown
        with st.container(border=True):
            st.markdown("<div class='section-title'>Confusion Matrix Breakdown (Test Set: 42,722 Total; 74 Fraud)</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Actual classification breakdown across True Positives, False Positives, False Negatives, and True Negatives.</div>", unsafe_allow_html=True)

            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                st.markdown("""
                <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:14px;'>
                    <b style='color:#0F172A; font-size:0.92rem;'>Random Forest</b><br>
                    <span style='font-size:0.84rem; color:#334155;'>
                    • <b>True Positives (TP):</b> 54<br>
                    • <b>False Positives (FP):</b> 5<br>
                    • <b>False Negatives (FN):</b> 20<br>
                    • <b>True Negatives (TN):</b> 42,643
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with cm2:
                st.markdown("""
                <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:14px;'>
                    <b style='color:#0F172A; font-size:0.92rem;'>Neural Network (ANN)</b><br>
                    <span style='font-size:0.84rem; color:#334155;'>
                    • <b>True Positives (TP):</b> 62<br>
                    • <b>False Positives (FP):</b> 46<br>
                    • <b>False Negatives (FN):</b> 12<br>
                    • <b>True Negatives (TN):</b> 42,602
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with cm3:
                st.markdown("""
                <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:14px;'>
                    <b style='color:#0F172A; font-size:0.92rem;'>Logistic Regression</b><br>
                    <span style='font-size:0.84rem; color:#334155;'>
                    • <b>True Positives (TP):</b> 63<br>
                    • <b>False Positives (FP):</b> 206<br>
                    • <b>False Negatives (FN):</b> 11<br>
                    • <b>True Negatives (TN):</b> 42,442
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with cm4:
                st.markdown("""
                <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:14px;'>
                    <b style='color:#0F172A; font-size:0.92rem;'>Autoencoder</b><br>
                    <span style='font-size:0.84rem; color:#334155;'>
                    • <b>True Positives (TP):</b> 52<br>
                    • <b>False Positives (FP):</b> 77<br>
                    • <b>False Negatives (FN):</b> 22<br>
                    • <b>True Negatives (TN):</b> 42,571
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        # Metric Definitions & Architecture Trade-offs (Collapsible to keep section uncluttered)
        with st.expander("Understanding Evaluation Metrics & Architecture Trade-offs", expanded=False):
            st.markdown("""
            - **Precision:** Proportion of flagged transactions that were actually fraudulent (True Positives / Predicted Positives). High precision minimizes false alarms for legitimate cardholders.
            - **Recall:** Proportion of actual fraud incidents successfully intercepted (True Positives / Actual Positives). High recall minimizes fraud leakage.
            - **F1-Score:** Harmonic mean of Precision and Recall, measuring overall classifier balance.
            - **PR-AUC:** Area under the Precision-Recall Curve; the most reliable performance metric on highly imbalanced datasets.
            - **ROC-AUC:** Area under the Receiver Operating Characteristic curve; measures class separation ability across varying decision thresholds.

            **Architecture Trade-offs & Scoring Differences:**
            - **Supervised Classifiers (Random Forest, ANN, Logistic Regression):** Trained on labeled records of both classes with SMOTE class balancing. These models output class-1 posterior probability scores between 0.0 and 1.0 (0% to 100%).
              - *Random Forest:* Delivers the highest precision (91.53%) with only 5 false positives out of 42,648 legitimate test transactions.
              - *Artificial Neural Network (ANN):* Achieves high sensitivity (83.78% recall, catching 62 of 74 test fraud cases) with 57.41% precision.
              - *Logistic Regression:* Provides an interpretable linear baseline with high recall (85.14%) and 206 false alarms (23.42% precision).
            - **Unsupervised Anomaly Detector (Autoencoder):** Trained strictly on legitimate transactions (Class 0) to learn normal transaction reconstruction. During inference, it computes a Mean Squared Error (MSE) reconstruction loss rather than a probability. Transactions with MSE exceeding the calibrated anomaly threshold (3.0607 MSE) are flagged as suspicious anomalies, achieving 70.27% recall without ever seeing fraud labels during training.
            """)


# =====================================================================
# PAGE 5: ABOUT
# =====================================================================
elif menu_selection == "About":
    st.markdown("""
    <div class='portal-topbar'>
        <div class='portal-title-box'>
            <h2 class='portal-title'>About FinSecure</h2>
            <p class='portal-sub'>Project overview, problem statement, system architecture, and academic specifications</p>
        </div>
        <div class='portal-badge'>
            ● ACADEMIC PBL SPECIFICATION
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='section-title'>Project Overview & Problem Statement</div>", unsafe_allow_html=True)
        st.markdown("""
        **FinSecure** is a credit card fraud surveillance platform engineered to evaluate transactions using both **Machine Learning** and **Deep Learning** algorithms.

        **The Core Problem:**
        - Credit card fraud causes billions of dollars in global annual losses.
        - Real-world fraud is an extreme **needle-in-a-haystack** problem: legitimate transactions outnumber fraud by over 500 to 1.
        - Detection systems must operate with sub-second latency while balancing two opposing requirements:
          1. **Catching as much fraud as possible** (High Recall).
          2. **Minimizing false declines for legitimate customers** (High Precision).
        """)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='section-title'>Dataset Information</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Source:** Machine Learning Group (ULB) / Kaggle European Credit Card Fraud Detection Benchmark.
        - **Records:** **284,807 transactions** (284,315 legitimate, 492 confirmed fraud).
        - **Timeline:** 48 hours of transaction records from European cardholders in September 2013.
        - **Features:** 30 inputs (`Time` in elapsed seconds, `V1` to `V28` anonymized PCA numerical components, `Amount` in Euros).
        - **Demonstration Currency:** Displays in Indian Rupees (₹) by default (1 EUR = ₹90.00). Model inference strictly runs on the original EUR scale.
        """)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='section-title'>System Workflow (4 Simple Steps)</div>", unsafe_allow_html=True)
        st.markdown("""
        1. **Data Ingestion:** Transaction parameters (`Time`, `V1`–`V28`, `Amount`) are received.
        2. **Normalization:** Input features are scaled using a pre-calibrated `RobustScaler` (based on median and IQR).
        3. **Pattern Analysis:** Supervised models compute fraud probabilities; the Autoencoder computes a reconstruction error (MSE).
        4. **Fraud Prediction:** The computed score is evaluated against calibrated decision thresholds to classify the transaction as **Legitimate** or **Potentially Fraudulent**.
        """)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='section-title'>Technology Stack</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Language:** Python 3.11
        - **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE), Joblib
        - **Deep Learning:** TensorFlow 2.18 / Keras (Multi-Layer Perceptron & Deep Autoencoder)
        - **Frontend & Visualization:** Streamlit, Plotly Express & Graph Objects
        - **Data Handling:** Pandas, NumPy
        """)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='section-title'>Academic Context & Project Disclaimer</div>", unsafe_allow_html=True)
        st.markdown("""
        This application was engineered for an undergraduate **B.Tech Computer Science & Engineering Project-Based Learning (PBL)** demonstration in Machine Learning and Deep Learning.

        **Educational Disclaimer:** FinSecure is an academic research and benchmarking platform built on historical dataset records. It is not connected to a live banking payment gateway or real-time card settlement network.
        """)
