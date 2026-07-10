import streamlit as st
import requests
import time
import pandas as pd

# --- Configuration & Layout ---
st.set_page_config(page_title="Device Telemetry Dashboard", layout="wide")

# GLOBAL LOCAL BACKEND URL
LOCAL_BACKEND_URL = "http://127.0.0.1:5000"

# --- Session State Initialization ---
if 'heap_history' not in st.session_state:
    st.session_state.heap_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Page Header ---
st.title("Device Telemetry Dashboard")
st.markdown("Monitoring local Flask backend.")

# --- Sidebar In-App AI Chatbot ---
st.sidebar.title("AI Assistant")
st.sidebar.markdown("Ask questions about the device status.")

# Render existing chat messages
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.sidebar.chat_input("Ask a question about the device..."):
    st.sidebar.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Match the /chatbot endpoint logic
        response = requests.post(f"{LOCAL_BACKEND_URL}/chatbot", json={"prompt": prompt}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("response", "No response received.")
            with st.sidebar.chat_message("assistant"):
                st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        else:
            st.sidebar.error(f"Backend Error: {response.status_code}")
    except requests.exceptions.RequestException:
        st.sidebar.error("Failed to connect to backend.")

# --- Metrics Display & Live Loop ---
st.subheader("Live Telemetry")

metrics_placeholder = st.empty()
chart_placeholder = st.empty()

while True:
    try:
        # Make GET request explicitly calling /get_live_data
        res = requests.get(f"{LOCAL_BACKEND_URL}/get_live_data", timeout=5)
        
        if res.status_code == 200:
            data = res.json()
            
            # Extract exactly the JSON keys returned by the backend
            vcc = data.get("vcc", 0.0)
            heap = data.get("heap", 0)
            failure_prob = data.get("failure_probability", 0.0)
            risk_status = data.get("risk_status", "UNKNOWN")
            
            st.session_state.heap_history.append(heap)
            if len(st.session_state.heap_history) > 100:
                st.session_state.heap_history.pop(0)

            # Render the 3-column metric display beautifully
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("Internal Voltage (VCC)", f"{vcc:.2f} V")
                col2.metric("Free Heap Memory", f"{heap} Bytes")
                
                # Dynamic rendering based on risk status
                if risk_status == "DANGER":
                    col3.metric("Failure Risk Probability", f"{failure_prob:.2f} %", delta="CRITICAL RISK", delta_color="inverse")
                elif risk_status == "WARNING":
                    col3.metric("Failure Risk Probability", f"{failure_prob:.2f} %", delta="WARNING", delta_color="off")
                else:
                    col3.metric("Failure Risk Probability", f"{failure_prob:.2f} %", delta="SAFE", delta_color="normal")
            
            with chart_placeholder.container():
                df = pd.DataFrame(st.session_state.heap_history, columns=['Heap Memory (Bytes)'])
                st.line_chart(df)
        else:
            with metrics_placeholder.container():
                st.warning(f"Waiting for backend data... HTTP {res.status_code}")
                
    except requests.exceptions.RequestException:
        with metrics_placeholder.container():
            st.error(f"Cannot reach Flask server at {LOCAL_BACKEND_URL}. Is it running?")

    time.sleep(1)
