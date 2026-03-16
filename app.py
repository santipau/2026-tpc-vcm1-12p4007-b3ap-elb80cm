import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

# ==========================================================
# MAIN APP
# ==========================================================
st.set_page_config(layout="wide")
st.title("Remaining Useful Life (RUL) Forecasting")

# ==========================================================
# DATA SELECTION
# ==========================================================
st.sidebar.header("Model Selection")

DATA_DIR = Path("data")

selected_dataset = "TPC_VCM1_12-P-4007-B3A-P_ELB80CM"
dataset_path = DATA_DIR / f"{selected_dataset}.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data(dataset_path)

st.sidebar.caption(f"Loaded: {selected_dataset}")


# ==========================================================
# CORROSION RATE STATISTICS
# ==========================================================
# Note: This section is added in app_rev2.py to calculate and display CR statistics before the simulation parameters.
st.subheader("Corrosion Rate (CR) Statistics")

window = 144*30  # 1 month of data at 6 samples/hour
hours = window / 6   # if 6 samples per hour

# Calculate CR statistics
recent_cr = df["PREDICTED_CR_MA"].dropna().tail(window)

if recent_cr.empty:
    st.warning("Not enough data.")
    st.stop()

mean_cr = -abs(recent_cr.mean())
std_cr = abs(recent_cr.std())

mean_cr_per_year = mean_cr*6*24*365
std_cr_per_year = std_cr*6*24*365

cr_worst = mean_cr - 1.96 * std_cr
cr_best  = mean_cr + 1.96 * std_cr

# Display metrics
# with right:
m1, m2 = st.columns(2)
m1.metric("Mean CR (mm/year)", f"{-mean_cr_per_year:.6f}")
m2.metric("Std Dev CR (mm/year)", f"{std_cr_per_year:.6f}")

# ==========================================================
# SIMULATION PARAMETERS
# ==========================================================
st.subheader("Simulation Parameters")

# Turnaround default dates depending on dataset
DEFAULT_1TA = pd.to_datetime("2028-09-01")
DEFAULT_2TA = pd.to_datetime("2031-01-01")

# Turnaround input
col1, col2 = st.columns(2)

with col1:
    turnaround_1st = st.date_input("1st Turnaround", value=DEFAULT_1TA.date())

with col2:
    turnaround_2nd = st.date_input("2nd Turnaround", value=DEFAULT_2TA.date())

# Convert to pandas timestamp
turnaround_1st = pd.Timestamp(turnaround_1st)
turnaround_2nd = pd.Timestamp(turnaround_2nd)

# Minimum thickness allowance
DEFAULT_MIN_THICKNESS = 4.0
min_thickness = st.number_input(
        "Minimum requirement thickness (mm)",
        value=DEFAULT_MIN_THICKNESS,
        step=0.1
    )   

# ==========================================================
# SIMULATION AND PLOTTING
# ==========================================================

# Ensure timestamp format
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Get last thickness and timestamp
last_thickness = df["PREDICTED_THICKNESS"].dropna().iloc[-1]
last_timestamp = pd.Timestamp(df["timestamp"].iloc[-1])

DAYS_PER_YEAR = 365


# ----------------------------------------------------------
# RUL Calculation
# ----------------------------------------------------------
def calculate_rul_days(cr):

    cr = abs(cr)
    remaining = last_thickness - min_thickness

    if cr <= 0:
        return np.inf

    if remaining <= 0:
        return 0

    loss_per_day = (cr / 10) * 60 * 24

    if loss_per_day <= 0:
        return np.inf

    return remaining / loss_per_day


# ----------------------------------------------------------
# Run Simulation
# ----------------------------------------------------------
if st.button("Forecasting"):

    rul_mean = calculate_rul_days(mean_cr)
    rul_worst = calculate_rul_days(cr_worst)
    rul_best = calculate_rul_days(cr_best)

    # ------------------------------------------------------
    # Forecast horizon based on RUL
    # ------------------------------------------------------
    max_days_rul = (
        int(max(rul_best, rul_worst) * 1.1)
        if np.isfinite(max(rul_best, rul_worst))
        else 365
    )

    # ------------------------------------------------------
    # Forecast horizon based on 2nd turnaround
    # ------------------------------------------------------
    days_to_ta2 = (pd.to_datetime(turnaround_2nd) - last_timestamp).days

    # Ensure forecast reaches the 2nd turnaround
    max_days = max(max_days_rul, days_to_ta2, 30)

    # ------------------------------------------------------
    # Future timeline
    # ------------------------------------------------------
    future_days = np.linspace(0, max_days, 300)
    future_dates = last_timestamp + pd.to_timedelta(future_days, unit="D")

    # ------------------------------------------------------
    # Forecast thickness
    # ------------------------------------------------------
    def forecast(cr):
        loss = (abs(cr) / 10) * 60 * 24
        return last_thickness - loss * future_days

    thk_mean = forecast(mean_cr)
    thk_worst = forecast(cr_worst)
    thk_best = forecast(cr_best)

    upper = np.maximum(thk_best, thk_worst)
    lower = np.minimum(thk_best, thk_worst)

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["PREDICTED_THICKNESS"],
        mode="lines",
        name="Historical",
        line=dict(width=2)
    ))

    # Mean Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=thk_mean,
        mode="lines",
        name="Mean Forecast",
        line=dict(dash="dash", width=2)
    ))

    # Confidence Band
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(200,0,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
        hoverinfo="skip"
    ))

    # Minimum Thickness
    fig.add_hline(
        y=min_thickness,
        line_dash="dot",
        annotation_text="Minimum Thickness",
        annotation_position="bottom right"
    )

    # ------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------
    ymin = min(df["PREDICTED_THICKNESS"].min(), np.min(lower))
    ymax = max(df["PREDICTED_THICKNESS"].max(), np.max(upper))

    # ------------------------------------------------------
    # Turnaround lines
    # ------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[turnaround_1st, turnaround_1st],
        y=[ymin, ymax],
        mode="lines",
        line=dict(color="green", dash="dash"),
        name="1st Turnaround"
    ))

    fig.add_trace(go.Scatter(
        x=[turnaround_2nd, turnaround_2nd],
        y=[ymin, ymax],
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name="2nd Turnaround"
    ))

    # ------------------------------------------------------
    # Annotations
    # ------------------------------------------------------

    fig.add_annotation(
        x=turnaround_1st,
        y=ymax,
        text="1st Turnaround",
        showarrow=False,
        yshift=10,
        font=dict(color="green")
    )

    fig.add_annotation(
        x=turnaround_2nd,
        y=ymax,
        text="2nd Turnaround",
        showarrow=False,
        yshift=10,
        font=dict(color="blue")
    )

    # ------------------------------------------------------
    # Layout
    # ------------------------------------------------------
    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date",
            range=[
                df["timestamp"].min(),
                max(future_dates[-1], pd.to_datetime(turnaround_2nd))
            ]
        ),
        yaxis=dict(title="Thickness"),
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # ------------------------------------------------------
    # Show Plot
    # ------------------------------------------------------
    st.subheader("Forecast Thickness with 95% Confidence Interval")
    st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------
    # RUL Display
    # ------------------------------------------------------
    def fmt(days):

        if np.isfinite(days) and days > 0:
            return f"{days:.1f} days", f"{days/DAYS_PER_YEAR:.2f} years"

        if days == 0:
            return "0 days", "Failure reached"

        return "∞", "No failure"


    col1, col2, col3 = st.columns(3)

    col1.metric("Worst Case", *fmt(rul_worst))
    col2.metric("Mean Case", *fmt(rul_mean))
    col3.metric("Best Case", *fmt(rul_best))


    # ------------------------------------------------------
    # Failure Date Estimation
    # ------------------------------------------------------
    if np.isfinite(rul_mean) and rul_mean > 0:

        failure_worst = last_timestamp + pd.Timedelta(days=float(rul_worst))
        failure_mean = last_timestamp + pd.Timedelta(days=float(rul_mean))
        failure_best = last_timestamp + pd.Timedelta(days=float(rul_best))

        d1, d2, d3 = st.columns(3)

        d1.metric("Worst Case Date", failure_worst.strftime("%Y-%m-%d"))
        d2.metric("Mean Case Date", failure_mean.strftime("%Y-%m-%d"))
        d3.metric("Best Case Date", failure_best.strftime("%Y-%m-%d"))


        # --------------------------------------------------
        # TA Recommendation
        # --------------------------------------------------
        st.markdown("<br><br>", unsafe_allow_html=True)
        if failure_worst <= turnaround_1st:
            st.markdown(
                "<h2 style='color:red;'>‼️ Failure expected BEFORE 1st Turnaround</h2>",
                unsafe_allow_html=True
            )
        elif failure_worst <= turnaround_2nd:
            st.markdown(
                "<h2 style='color:orange;'>⚠️ Failure expected BEFORE 2nd Turnaround</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:green;'>✅ Safe until next planned turnaround</h2>",
                unsafe_allow_html=True
            )

    elif rul_mean == 0:
        st.error("Component already below minimum thickness!")
    else:
        st.success("No predicted failure (CR ≈ 0)")# Turnaround input
col1, col2 = st.columns(2)

with col1:
    turnaround_1st = st.date_input("1st Turnaround", value=DEFAULT_1TA.date())

with col2:
    turnaround_2nd = st.date_input("2nd Turnaround", value=DEFAULT_2TA.date())

# Convert to pandas timestamp
turnaround_1st = pd.Timestamp(turnaround_1st)
turnaround_2nd = pd.Timestamp(turnaround_2nd)

# Minimum thickness allowance
DEFAULT_MIN_THICKNESS = 3.0
min_thickness = st.number_input(
        "Minimum requirement thickness (mm)",
        value=DEFAULT_MIN_THICKNESS,
        step=0.1
    )   

# ==========================================================
# SIMULATION AND PLOTTING
# ==========================================================

# Ensure timestamp format
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Get last thickness and timestamp
last_thickness = df["PREDICTED_THICKNESS"].dropna().iloc[-1]
last_timestamp = pd.Timestamp(df["timestamp"].iloc[-1])

DAYS_PER_YEAR = 365


# ----------------------------------------------------------
# RUL Calculation
# ----------------------------------------------------------
def calculate_rul_days(cr):

    cr = abs(cr)
    remaining = last_thickness - min_thickness

    if cr <= 0:
        return np.inf

    if remaining <= 0:
        return 0

    loss_per_day = (cr / 10) * 60 * 24

    if loss_per_day <= 0:
        return np.inf

    return remaining / loss_per_day


# ----------------------------------------------------------
# Run Simulation
# ----------------------------------------------------------
if st.button("Forecasting"):

    rul_mean = calculate_rul_days(mean_cr)
    rul_worst = calculate_rul_days(cr_worst)
    rul_best = calculate_rul_days(cr_best)

    # ------------------------------------------------------
    # Forecast horizon based on RUL
    # ------------------------------------------------------
    max_days_rul = (
        int(max(rul_best, rul_worst) * 1.1)
        if np.isfinite(max(rul_best, rul_worst))
        else 365
    )

    # ------------------------------------------------------
    # Forecast horizon based on 2nd turnaround
    # ------------------------------------------------------
    days_to_ta2 = (pd.to_datetime(turnaround_2nd) - last_timestamp).days

    # Ensure forecast reaches the 2nd turnaround
    max_days = max(max_days_rul, days_to_ta2, 30)

    # ------------------------------------------------------
    # Future timeline
    # ------------------------------------------------------
    future_days = np.linspace(0, max_days, 300)
    future_dates = last_timestamp + pd.to_timedelta(future_days, unit="D")

    # ------------------------------------------------------
    # Forecast thickness
    # ------------------------------------------------------
    def forecast(cr):
        loss = (abs(cr) / 10) * 60 * 24
        return last_thickness - loss * future_days

    thk_mean = forecast(mean_cr)
    thk_worst = forecast(cr_worst)
    thk_best = forecast(cr_best)

    upper = np.maximum(thk_best, thk_worst)
    lower = np.minimum(thk_best, thk_worst)

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["PREDICTED_THICKNESS"],
        mode="lines",
        name="Historical",
        line=dict(width=2)
    ))

    # Mean Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=thk_mean,
        mode="lines",
        name="Mean Forecast",
        line=dict(dash="dash", width=2)
    ))

    # Confidence Band
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(200,0,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
        hoverinfo="skip"
    ))

    # Minimum Thickness
    fig.add_hline(
        y=min_thickness,
        line_dash="dot",
        annotation_text="Minimum Thickness",
        annotation_position="bottom right"
    )

    # ------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------
    ymin = min(df["PREDICTED_THICKNESS"].min(), np.min(lower))
    ymax = max(df["PREDICTED_THICKNESS"].max(), np.max(upper))

    # ------------------------------------------------------
    # Turnaround lines
    # ------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[turnaround_1st, turnaround_1st],
        y=[ymin, ymax],
        mode="lines",
        line=dict(color="green", dash="dash"),
        name="1st Turnaround"
    ))

    fig.add_trace(go.Scatter(
        x=[turnaround_2nd, turnaround_2nd],
        y=[ymin, ymax],
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name="2nd Turnaround"
    ))

    # ------------------------------------------------------
    # Annotations
    # ------------------------------------------------------

    fig.add_annotation(
        x=turnaround_1st,
        y=ymax,
        text="1st Turnaround",
        showarrow=False,
        yshift=10,
        font=dict(color="green")
    )

    fig.add_annotation(
        x=turnaround_2nd,
        y=ymax,
        text="2nd Turnaround",
        showarrow=False,
        yshift=10,
        font=dict(color="blue")
    )

    # ------------------------------------------------------
    # Layout
    # ------------------------------------------------------
    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date",
            range=[
                df["timestamp"].min(),
                max(future_dates[-1], pd.to_datetime(turnaround_2nd))
            ]
        ),
        yaxis=dict(title="Thickness"),
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # ------------------------------------------------------
    # Show Plot
    # ------------------------------------------------------
    st.subheader("Forecast Thickness with 95% Confidence Interval")
    st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------
    # RUL Display
    # ------------------------------------------------------
    def fmt(days):

        if np.isfinite(days) and days > 0:
            return f"{days:.1f} days", f"{days/DAYS_PER_YEAR:.2f} years"

        if days == 0:
            return "0 days", "Failure reached"

        return "∞", "No failure"


    col1, col2, col3 = st.columns(3)

    col1.metric("Worst Case", *fmt(rul_worst))
    col2.metric("Base Case", *fmt(rul_mean))
    col3.metric("Best Case", *fmt(rul_best))


    # ------------------------------------------------------
    # Failure Date Estimation
    # ------------------------------------------------------
    if np.isfinite(rul_mean) and rul_mean > 0:

        failure_worst = last_timestamp + pd.Timedelta(days=float(rul_worst))
        failure_mean = last_timestamp + pd.Timedelta(days=float(rul_mean))
        failure_best = last_timestamp + pd.Timedelta(days=float(rul_best))

        d1, d2, d3 = st.columns(3)

        d1.metric("Worst Case Date", failure_worst.strftime("%Y-%m-%d"))
        d2.metric("Base Case Date", failure_mean.strftime("%Y-%m-%d"))
        d3.metric("Best Case Date", failure_best.strftime("%Y-%m-%d"))


        # --------------------------------------------------
        # TA Recommendation
        # --------------------------------------------------
        st.markdown("<br><br>", unsafe_allow_html=True)
        if failure_worst <= turnaround_1st:
            st.markdown(
                "<h2 style='color:red;'>‼️ Failure expected BEFORE 1st Turnaround</h2>",
                unsafe_allow_html=True
            )
        elif failure_worst <= turnaround_2nd:
            st.markdown(
                "<h2 style='color:orange;'>⚠️ Failure expected BEFORE 2nd Turnaround</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:green;'>✅ Safe until next planned turnaround</h2>",
                unsafe_allow_html=True
            )

    elif rul_mean == 0:
        st.error("Component already below minimum thickness!")
    else:
        st.success("No predicted failure (CR ≈ 0)")
