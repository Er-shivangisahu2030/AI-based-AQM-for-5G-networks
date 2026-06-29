import streamlit as st
import requests

st.set_page_config(
    page_title="AI-Based AQM for 5G Networks",
    page_icon="📡",
    layout="wide"
)

st.title("AI-Based Active Queue Management for 5G Networks")

st.write("Enter network parameters to predict congestion.")

col1, col2 = st.columns(2)

with col1:

    queue_length = st.number_input(
        "Queue Length",
        min_value=0.0,
        value=10.0
    )

    arrival_rate = st.number_input(
        "Arrival Rate",
        min_value=0.0,
        value=5.0
    )

    service_rate = st.number_input(
        "Service Rate",
        min_value=0.0,
        value=4.0
    )

with col2:

    avg_delay_ms = st.number_input(
        "Average Delay (ms)",
        min_value=0.0,
        value=2.0
    )

    throughput = st.number_input(
        "Throughput",
        min_value=0.0,
        value=5.0
    )

    drop_rate = st.number_input(
        "Drop Rate",
        min_value=0.0,
        value=0.0
    )

if st.button("Predict Congestion"):

    payload = {

        "queue_length": queue_length,
        "arrival_rate": arrival_rate,
        "service_rate": service_rate,
        "avg_delay_ms": avg_delay_ms,
        "throughput": throughput,
        "drop_rate": drop_rate

    }

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        pred = result["prediction"]
        prob = result["probability"]

        if pred == 1:
            st.error("Network Congested")
        else:
            st.success("Network Normal")

        st.metric(
            "Congestion Probability",
            f"{prob*100:.2f}%"
        )

    else:
        st.error("Flask server is not running.")