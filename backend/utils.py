import pickle
import os

def load_model(model_path="model.pkl"):
    """
    Loads the Random Forest model from disk.
    Returns: (model_object, message_string)
    """
    if not os.path.exists(model_path):
        return None, "Model file not found."
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, "Model loaded successfully."
    except Exception as e:
        return None, f"Error loading model: {e}"

def calculate_risk_status(probability):
    """
    Calculates risk status based on the predicted failure probability.
    <30% -> SAFE
    30-70% -> WARNING
    >70% -> DANGER
    """
    if probability < 30.0:
        return "SAFE"
    elif probability <= 70.0:
        return "WARNING"
    else:
        return "DANGER"
