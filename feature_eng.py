from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    return accuracy, precision, recall, f1, report

def get_diabetes_risk_message(prediction):
    if prediction == 0:
        return "Low Risk (Non-Diabetic)", "Congratulations! Based on the provided features, the individual is not predicted to be at risk of diabetes. Maintaining a healthy lifestyle is always beneficial."
    elif prediction == 1:
        return "Moderate Risk", "Based on the provided features, the individual is predicted to have a moderate risk of diabetes. We recommend monitoring lifestyle factors such as diet and exercise to reduce the risk."
    else:
        return "High Risk (Diabetic)", "Alert: Based on the provided features, the individual is predicted to be at high risk of diabetes. We strongly recommend consulting a healthcare professional for further evaluation and advice."
