import pandas as pd
import joblib

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load a saved model from a file."""
    return joblib.load(filepath)
