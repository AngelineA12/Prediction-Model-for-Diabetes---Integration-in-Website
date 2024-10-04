import joblib
import pandas as pd

def save_model(model, file_path):
    """
    Save the trained model for future use.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Load a trained model from file.
    """
    return joblib.load(file_path)

def predict_diabetes_risk(model, feature_values):
    input_data = pd.DataFrame([feature_values])
    prediction = model.predict(input_data)
    return prediction[0]
