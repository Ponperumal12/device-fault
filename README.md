# Smart Electronic Devices Failure Prediction Platform

This repository contains the backend and dashboard for predicting device failures based on internal voltage and memory usage.

## Project Structure
```text
device_fault/
├── backend/
│   ├── flask_backend.py        # Flask API server providing the prediction API
│   ├── device_model.pkl        # Trained Random Forest ML model
│   ├── requirements.txt        # Python dependencies
│   └── utils.py                # Helper utilities
│
├── dashboard/
│   └── streamlit_app.py        # Streamlit UI dashboard
│
└── README.md                   # Project summary
```

## Running the Application

**1. Start the Backend API (Terminal 1)**
```bash
cd backend
pip install -r requirements.txt
python flask_backend.py
```
*Runs locally on `http://127.0.0.1:5000`*

**2. Start the Dashboard (Terminal 2)**
```bash
cd dashboard
streamlit run streamlit_app.py
```
