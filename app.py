import streamlit as st
import pickle
import time
import random
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Electronic Devices Failure Prediction Platform",
    page_icon="⚡",
    layout="wide"
)

# --- ML Model Import & Safety ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model."""
    model_path = 'model.pkl'
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return None
    except Exception as e:
        return e

# Try to load the model
model = load_model()

if model is None:
    st.sidebar.warning("⚠️ 'model.pkl' not found in the current directory. Running in simulation mode without ML predictions.")
elif isinstance(model, Exception):
    st.sidebar.error(f"❌ Error loading 'model.pkl': {model}")
    model = None
else:
    st.sidebar.success("✅ ML Model 'model.pkl' loaded successfully.")

# --- Sidebar: AI Chatbot (Dual-Language) ---
st.sidebar.header("🤖 Assistant (AI Chatbot)")
st.sidebar.markdown("Ask about device status in English or Tamil.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today? / வணக்கம்! நான் உங்கள் ஏஐ (AI) உதவியாளர். நான் உங்களுக்கு எப்படி உதவ முடியும்?"}
    ]

# Display chat history in sidebar
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input processing
if prompt := st.sidebar.chat_input("Type 'status' or 'நிலை'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower()
    current_risk = st.session_state.get('current_risk', 0.0)
    
    # Conditional/rule-based responses
    if "status" in prompt_lower or "நிலை" in prompt_lower:
        if current_risk > 80:
            response = f"🚨 CRITICAL DANGER: Failure risk is {current_risk:.2f}%. Immediate action required! / 🚨 மிக ஆபத்து: தோல்வி அபாயம் {current_risk:.2f}%. உடனடி நடவடிக்கை தேவை!"
        elif current_risk > 50:
            response = f"⚠️ WARNING: Failure risk is elevated at {current_risk:.2f}%. / ⚠️ எச்சரிக்கை: தோல்வி அபாயம் {current_risk:.2f}% ஆக உயர்ந்துள்ளது."
        else:
            response = f"✅ NORMAL: System is stable. Failure risk is {current_risk:.2f}%. / ✅ இயல்பானது: சிஸ்டம் நிலையாக உள்ளது. தோல்வி அபாயம் {current_risk:.2f}%."
    else:
        response = "I can only answer about 'status' right now. / தற்போது 'status' அல்லது 'நிலை' பற்றி மட்டுமே என்னால் பதிலளிக்க முடியும்."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.sidebar.chat_message("assistant"):
        st.markdown(response)

# --- Main Dashboard ---
st.title("⚡ Smart Electronic Devices Failure Prediction Platform")
st.markdown("Monitor internal hardware telemetry in real-time and predict device failure.")
st.markdown("---")

# Layout placeholders
metrics_placeholder = st.empty()
chart_placeholder = st.empty()

# Simulation State Initialization
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Free_Heap_Memory"])
if 'tick' not in st.session_state:
    st.session_state.tick = 0
if 'simulating' not in st.session_state:
    st.session_state.simulating = False
if 'current_risk' not in st.session_state:
    st.session_state.current_risk = 0.0

# Simulation Controls
st.write("### Simulation Controls")
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
if col_btn1.button("▶️ Start Live Simulation"):
    st.session_state.simulating = True
if col_btn2.button("⏹️ Stop Simulation"):
    st.session_state.simulating = False

# --- Live Telemetry Simulation Loop ---
if st.session_state.simulating:
    # Use a loop with a fixed number of iterations to avoid complete blocking of Streamlit
    # The user requested a continuous real-time rendering loop with st.empty() and time.sleep(1)
    
    for _ in range(100):  # Run for 100 seconds max per click to keep app responsive eventually
        if not st.session_state.simulating:
            break
            
        st.session_state.tick += 1
        tick = st.session_state.tick
        
        # 1. Generate Telemetry
        vcc = round(random.uniform(3.1, 3.3), 2)
        
        # Fault Simulation Mode (Trigger artificial memory leak around tick 15)
        if 15 <= tick <= 30:
            free_heap = random.randint(4500, 5500) # Drastic drop
        else:
            free_heap = random.randint(45000, 52000) # Normal

        # 2. Predict Failure Risk
        risk_prob = 0.0
        if model is not None:
            try:
                # Model expects [[VCC, Free_Heap]]
                features = [[vcc, free_heap]]
                prediction = model.predict_proba(features)
                risk_prob = prediction[0][1] * 100
            except Exception as e:
                # Fallback if prediction fails for some reason
                risk_prob = 98.5 if free_heap < 10000 else 2.5
        else:
            # Mock prediction logic if model is missing (for hackathon demo purposes)
            risk_prob = 98.5 if free_heap < 10000 else random.uniform(1.0, 5.0)
            
        st.session_state.current_risk = risk_prob

        # Update History for Chart
        new_row = pd.DataFrame({"Time": [tick], "Free_Heap_Memory": [free_heap]})
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        
        # Keep last 50 points to maintain a clean scrolling chart
        if len(st.session_state.history) > 50:
            st.session_state.history = st.session_state.history.iloc[-50:]

        # 3. Render Metrics dynamically
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Internal Voltage (VCC)", 
                    value=f"{vcc} V", 
                    delta="Stable" if vcc >= 3.1 else "Low"
                )
                
            with col2:
                # Changing delta to red inverse when memory drops drastically
                if free_heap < 10000:
                    heap_delta = "-45000 (CRITICAL LEAK)"
                    delta_color = "inverse"
                else:
                    heap_delta = "Normal"
                    delta_color = "normal"
                    
                st.metric(
                    label="Free Heap Memory (Bytes)", 
                    value=f"{free_heap:,}", 
                    delta=heap_delta, 
                    delta_color=delta_color
                )
                
            with col3:
                # Changing delta to red inverse when risk spikes
                if risk_prob > 80:
                    risk_delta = "CRITICAL / DANGER"
                    r_delta_color = "inverse"
                elif risk_prob > 50:
                    risk_delta = "Warning"
                    r_delta_color = "inverse"
                else:
                    risk_delta = "Safe"
                    r_delta_color = "normal"
                    
                st.metric(
                    label="Failure Risk Probability", 
                    value=f"{risk_prob:.1f}%", 
                    delta=risk_delta, 
                    delta_color=r_delta_color
                )

        # 4. Render Chart dynamically
        with chart_placeholder.container():
            st.subheader("Historical Trend: Free Heap Memory")
            chart_data = st.session_state.history.set_index("Time")
            st.line_chart(chart_data)

        # 5. Sleep for real-time simulation update
        time.sleep(1)
        
else:
    # Render static initial state when not simulating
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Internal Voltage (VCC)", "---")
        col2.metric("Free Heap Memory (Bytes)", "---")
        col3.metric("Failure Risk Probability", "---")
    
    with chart_placeholder.container():
        st.subheader("Historical Trend: Free Heap Memory")
        st.info("Click '▶️ Start Live Simulation' to begin streaming telemetry data.")
