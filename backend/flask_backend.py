"""
Smart Electronic Devices Failure Prediction Platform
Flask Backend API Server
Author: Backend Engine

Features:
  - Random Forest ML model prediction via /telemetry
  - Live data store served to Streamlit via /get_live_data
  - Rule-based chatbot via /chatbot
  - Telegram Bot alert notification (fires ONCE on DANGER, resets on SAFE)
"""

import os
import time
import pickle
import logging
import requests as http_requests  # renamed to avoid clash with Flask's request object
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# LOGGING SETUP - All activity visible in VS Code terminal
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes so Streamlit UI never gets blocked
CORS(app)

# ============================================================
# 1. DYNAMIC MODEL LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for model in backend/ first, then one level up in root
_model_candidates = [
    os.path.join(BASE_DIR, 'device_model.pkl'),           # backend/device_model.pkl
    os.path.join(BASE_DIR, '..', 'device_model.pkl'),     # root/device_model.pkl
]

loaded_model = None
MODEL_PATH_USED = None

for candidate in _model_candidates:
    _resolved = os.path.normpath(candidate)
    if os.path.exists(_resolved):
        try:
            with open(_resolved, 'rb') as f:
                loaded_model = pickle.load(f)
            MODEL_PATH_USED = _resolved
            log.info(f"[STARTUP SUCCESS] ✅ Model loaded from: {MODEL_PATH_USED}")
            break
        except Exception as e:
            log.error(f"[STARTUP ERROR] ❌ Found model at {_resolved} but failed to load it: {e}")
            break

if loaded_model is None and MODEL_PATH_USED is None:
    log.warning(
        "[STARTUP WARNING] ⚠️  device_model.pkl NOT FOUND.\n"
        f"  Searched in:\n"
        f"    1. {os.path.normpath(_model_candidates[0])}\n"
        f"    2. {os.path.normpath(_model_candidates[1])}\n"
        "  Please copy device_model.pkl to one of those locations.\n"
        "  Server is running, but /telemetry predictions will return a 503 error."
    )

# ============================================================
# 2. TELEGRAM BOT CREDENTIALS
# ============================================================
# WARNING: Keep these secret. Do not commit to public Git repos.
TELEGRAM_BOT_TOKEN = "8804524919:AAEmTlQaZ70pGvHJbYLbSvg1C3H1sybAQ9A"
TELEGRAM_CHAT_ID   = "6829491824"

# Alert state flag — ensures the notification fires ONLY ONCE per DANGER event.
# Automatically resets to False when device returns to SAFE status.
telegram_alert_sent = False

# ============================================================
# 3. IN-MEMORY GLOBAL DATA STORE (no database needed)
# ============================================================
live_device_data = {
    "vcc": 0.0,
    "heap": 0,
    "failure_probability": 0.0,
    "risk_status": "UNKNOWN",
    "timestamp": "Waiting for first telemetry..."
}

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def classify_risk(probability: float) -> str:
    """Classify failure probability into a human-readable risk level."""
    if probability < 30.0:
        return "SAFE"
    elif probability <= 70.0:
        return "WARNING"
    else:
        return "DANGER"


def send_telegram_notification(vcc: float, heap: float, probability: float) -> None:
    """
    Sends a bilingual (English/Tamil) Markdown-formatted Telegram alert
    to the configured TELEGRAM_CHAT_ID when device enters DANGER state.
    Runs in a fire-and-forget fashion — errors are logged but do not crash the server.
    """
    message = (
        "*[DEVICE ALERT / சாதன எச்சரிக்கை]*\n"
        "\n"
        "*Status / நிலை:* DANGER (அபாயம்)\n"
        f"*Voltage / மின்னழுத்தம்:* `{vcc:.2f} V`\n"
        f"*Heap Memory / நினைவகம்:* `{heap:.0f} bytes`\n"
        f"*Failure Probability / செயலிழப்பு வாய்ப்பு:* `{probability:.2f}%`\n"
        "\n"
        "*Action Required / நடவடிக்கை தேவை:*\n"
        "EN: Inspect the device immediately. Check power supply and firmware memory usage.\n"
        "TA: சாதனத்தை உடனே சரிபாருங்கள். மின்சாரம் மற்றும் நினைவக பயன்பாட்டை சரிபாருங்கள்."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = http_requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            log.info("[TELEGRAM] Alert sent successfully to chat_id: {}".format(TELEGRAM_CHAT_ID))
        else:
            log.warning("[TELEGRAM] Failed to send alert. Status: {}, Response: {}".format(
                response.status_code, response.text
            ))
    except http_requests.exceptions.RequestException as e:
        log.error("[TELEGRAM] Network error while sending alert: {}".format(e))


def build_chatbot_fallback(prompt: str) -> str:
    """
    Rule-based chatbot fallback used when Gemini API key is unavailable.
    Responds using live_device_data context.
    """
    p = prompt.lower()
    vcc   = live_device_data.get("vcc", "N/A")
    heap  = live_device_data.get("heap", "N/A")
    prob  = live_device_data.get("failure_probability", "N/A")
    risk  = live_device_data.get("risk_status", "UNKNOWN")
    ts    = live_device_data.get("timestamp", "N/A")

    if any(k in p for k in ["voltage", "vcc"]):
        return f"Current internal voltage (VCC) is {vcc}V (last updated: {ts})."
    elif any(k in p for k in ["heap", "memory"]):
        return f"Free heap memory is currently {heap} bytes (last updated: {ts})."
    elif any(k in p for k in ["safe", "status", "risk", "danger"]):
        return (
            f"The device status is '{risk}' with a failure probability of {prob}%. "
            f"Last updated: {ts}."
        )
    elif any(k in p for k in ["why", "reason", "cause", "high"]):
        return (
            "High failure risk is usually caused by abnormal voltage fluctuations (VCC too low or too high) "
            "or critically low free heap memory, which can indicate a firmware memory leak."
        )
    elif any(k in p for k in ["fix", "solve", "repair", "memory leak"]):
        return (
            "To fix a memory leak: review your firmware for unfreed dynamic allocations, "
            "avoid recursive functions that grow the stack, and restart the device if heap drops below 10,000 bytes."
        )
    elif any(k in p for k in ["red", "led"]):
        return (
            "A red LED means the device is in DANGER state (failure probability > 70%). "
            "Inspect the power supply and firmware immediately."
        )
    elif any(k in p for k in ["tamil", "தமிழ்"]):
        return (
            f"சாதன நிலை: '{risk}'. "
            f"மின்னழுத்தம்: {vcc}V. "
            f"Heap நினைவகம்: {heap} bytes. "
            f"செயலிழப்பு வாய்ப்பு: {prob}%."
        )
    else:
        return (
            f"I'm your Device AI Assistant. Device is currently '{risk}' "
            f"(VCC: {vcc}V, Heap: {heap} bytes, Failure Probability: {prob}%). "
            "Ask me about voltage, memory, risk status, or how to fix issues."
        )

# ============================================================
# 5. ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    log.warning(f"[404] Unknown route accessed: {request.path}")
    return jsonify({"status": "error", "message": f"Route '{request.path}' not found on this server."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    log.warning(f"[405] Wrong HTTP method on route: {request.path}")
    return jsonify({"status": "error", "message": f"Method '{request.method}' not allowed for '{request.path}'."}), 405

@app.errorhandler(500)
def internal_error(e):
    log.error(f"[500] Internal server error: {e}")
    return jsonify({"status": "error", "message": "Internal server error. Check the VS Code terminal for details."}), 500

# ============================================================
# 6. API ENDPOINTS
# ============================================================

# ------ Health Check ------
@app.route('/status', methods=['GET'])
def status():
    """Simple health check to confirm server and model are alive."""
    log.info("[GET /status] Health check called.")
    return jsonify({
        "server": "running",
        "model": "loaded" if loaded_model else "NOT_LOADED",
        "model_path": MODEL_PATH_USED or "Not found"
    }), 200


# ------ API 1: Receive Device Telemetry ------
@app.route('/telemetry', methods=['POST'])
def telemetry():
    """
    POST /telemetry
    Body:   {"vcc": 3.3, "heap": 45000}
    Return: {"status": "success", "failure_probability": 8.2, "risk_status": "SAFE"}
    """
    log.info(f"[POST /telemetry] Request received from {request.remote_addr}")

    # Enforce Content-Type: application/json
    if not request.is_json:
        log.warning("[POST /telemetry] ❌ Rejected: Content-Type is not application/json")
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        log.warning("[POST /telemetry] ❌ Rejected: Empty or malformed JSON body")
        return jsonify({"status": "error", "message": "Request body is empty or not valid JSON"}), 400

    # Validate keys
    vcc_raw  = data.get('vcc')
    heap_raw = data.get('heap')

    if vcc_raw is None or heap_raw is None:
        log.warning(f"[POST /telemetry] ❌ Missing keys. Received keys: {list(data.keys())}")
        return jsonify({"status": "error", "message": "Missing required keys: 'vcc' and/or 'heap'"}), 400

    # Validate types
    try:
        vcc  = float(vcc_raw)
        heap = float(heap_raw)
    except (ValueError, TypeError):
        log.warning(f"[POST /telemetry] ❌ Non-numeric values: vcc={vcc_raw}, heap={heap_raw}")
        return jsonify({"status": "error", "message": "'vcc' and 'heap' must be numeric values"}), 400

    # Validate logical range
    if vcc < 0 or heap < 0:
        log.warning(f"[POST /telemetry] ❌ Negative values: vcc={vcc}, heap={heap}")
        return jsonify({"status": "error", "message": "'vcc' and 'heap' cannot be negative"}), 400

    log.info(f"[POST /telemetry] ✅ Valid data received → vcc={vcc}V, heap={heap} bytes")

    # Check model
    if loaded_model is None:
        log.error("[POST /telemetry] ❌ Model is not loaded. Cannot run prediction.")
        return jsonify({"status": "error", "message": "ML model is not loaded. Place device_model.pkl in the backend or root folder."}), 503

    try:
        input_array = np.array([[vcc, heap]])
        probabilities = loaded_model.predict_proba(input_array)[0]

        # Class index 1 = failure class
        failure_prob = round(float(probabilities[1] if len(probabilities) > 1 else probabilities[0]) * 100, 2)
        risk_status  = classify_risk(failure_prob)

        # Update global in-memory store
        live_device_data["vcc"]                 = vcc
        live_device_data["heap"]                = heap
        live_device_data["failure_probability"] = failure_prob
        live_device_data["risk_status"]          = risk_status
        live_device_data["timestamp"]            = time.ctime()

        log.info("[POST /telemetry] Prediction: probability={}%, risk={}".format(failure_prob, risk_status))

        # ── TELEGRAM ALERT LOGIC ─────────────────────────────────────────────
        global telegram_alert_sent

        if risk_status == "DANGER" and not telegram_alert_sent:
            # Device just entered DANGER zone — fire the alert exactly once
            log.warning("[TELEGRAM] DANGER detected! Sending Telegram alert...")
            send_telegram_notification(vcc, heap, failure_prob)
            telegram_alert_sent = True  # Arm the flag to prevent spam

        elif risk_status == "SAFE" and telegram_alert_sent:
            # Device recovered to SAFE — reset flag so next DANGER triggers again
            log.info("[TELEGRAM] Device returned to SAFE. Alert flag reset.")
            telegram_alert_sent = False
        # ── END TELEGRAM ALERT LOGIC ─────────────────────────────────────────

        return jsonify({
            "status": "success",
            "failure_probability": failure_prob,
            "risk_status": risk_status
        }), 200

    except Exception as e:
        log.exception("[POST /telemetry] Prediction crashed: {}".format(e))
        return jsonify({"status": "error", "message": "Prediction failed: {}".format(str(e))}), 500


# ------ API 2: Return Live Data to Streamlit ------
@app.route('/get_live_data', methods=['GET'])
def get_live_data():
    """
    GET /get_live_data
    Returns the full in-memory telemetry dictionary to the Streamlit dashboard.
    """
    log.info(f"[GET /get_live_data] Returning live data → {live_device_data}")
    return jsonify(live_device_data), 200


# ------ API 3: Chatbot ------
@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    POST /chatbot
    Body:   {"prompt": "Why is the risk high?"}
    Return: {"status": "success", "response": "..."}
    """
    log.info(f"[POST /chatbot] Request received from {request.remote_addr}")

    if not request.is_json:
        log.warning("[POST /chatbot] ❌ Rejected: Content-Type is not application/json")
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Empty or malformed JSON body"}), 400

    prompt = data.get('prompt', '').strip()
    if not prompt:
        log.warning("[POST /chatbot] ❌ Empty prompt received")
        return jsonify({"status": "error", "message": "'prompt' field is required and cannot be empty"}), 400

    log.info(f"[POST /chatbot] 💬 Prompt received: '{prompt}'")

    # ── Optional: Swap this block for real Gemini API integration ──────────
    # try:
    #     import google.generativeai as genai
    #     genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    #     model = genai.GenerativeModel("gemini-1.5-flash")
    #     context = f"Device Status: {live_device_data}. User Question: {prompt}"
    #     response_text = model.generate_content(context).text
    # except Exception as e:
    #     log.warning(f"[CHATBOT] Gemini failed ({e}), using fallback logic.")
    #     response_text = build_chatbot_fallback(prompt)
    # ── End Gemini Block ────────────────────────────────────────────────────

    # Rule-based fallback (always works without any API key)
    response_text = build_chatbot_fallback(prompt)

    log.info(f"[POST /chatbot] ✅ Response: '{response_text[:80]}...'")
    return jsonify({
        "status": "success",
        "response": response_text
    }), 200


# ============================================================
# 7. RUN SERVER
# ============================================================
if __name__ == '__main__':
    log.info("=" * 60)
    log.info("  Smart Device Failure Prediction — Flask Backend")
    log.info("  Listening on http://0.0.0.0:5000")
    log.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
