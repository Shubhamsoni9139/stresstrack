import streamlit as st
import pandas as pd
import requests
import random
import time
import os
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "live_training_data.csv"

st.set_page_config(page_title="Stress Prediction Dashboard", layout="wide")
st.title("📊 Real-Time Stress Prediction Dashboard")


# Function to simulate input data
def generate_input():
    return {
        "HR": round(random.uniform(60, 140), 2),
        "respr": round(random.uniform(6, 20), 2),
        "Time_sec": int(time.time()),
    }


# Send data and get prediction
def send_and_receive():
    data = generate_input()

    with st.container():
        st.markdown("### 📤 Sending New Input")
        st.json(data)

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.success("✅ **Model Predictions**")
                st.json(result["predictions"])

            with col2:
                st.success("🎯 **Model Accuracies**")
                st.metric("Random Forest", result["accuracies"]["random_forest"])
                st.metric(
                    "Logistic Regression", result["accuracies"]["logistic_regression"]
                )
                st.metric("Ensemble", result["accuracies"]["ensemble"])

        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"🔌 Connection failed: {e}")


# Load data from CSV
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


# Show last 5 predictions
def show_last_predictions(df):
    st.subheader("🕒 Last 5 Predictions")
    st.dataframe(df.tail(5).reset_index(drop=True), use_container_width=True)


# Show analytics
def show_analytics(df):
    st.subheader("📈 Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### HR & Respiration over Time")
        fig = px.line(
            df.tail(50),
            x="Time_sec",
            y=["HR", "respr"],
            title="HR and Respiration Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Model Predictions Scatter")
        for model in ["random_forest", "logistic_regression", "ensemble"]:
            fig = px.scatter(
                df.tail(100), x="HR", y=model, color=model, title=f"{model} vs HR"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Model Votes Distribution")
    vote_counts = (
        df[["random_forest", "logistic_regression", "ensemble"]]
        .apply(pd.Series.value_counts)
        .fillna(0)
        .T
    )
    vote_counts.columns = ["Class 0", "Class 1"]
    st.bar_chart(vote_counts)

    st.markdown("### 🔥 Prediction Trends")
    fig = px.area(
        df.tail(100),
        x="Time_sec",
        y=["random_forest", "logistic_regression", "ensemble"],
        title="Model Prediction Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔄 Correlation Between Models")
    st.dataframe(df[["random_forest", "logistic_regression", "ensemble"]].corr())


# Auto-run logic
send_and_receive()
df = load_data()
if not df.empty:
    show_last_predictions(df)
    show_analytics(df)

# Auto-refresh every 3 seconds
st.experimental_set_query_params(run="true")
time.sleep(3)
st.rerun()
