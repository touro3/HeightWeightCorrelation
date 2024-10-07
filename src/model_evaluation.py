import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For loading the saved model

def load_model(filepath):
    """Load the saved model."""
    return joblib.load(filepath)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("---- Model Evaluation ----")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
    plt.xlabel('Actual Weight (Standardized)')
    plt.ylabel('Predicted Weight (Standardized)')
    plt.title('Actual vs Predicted Weight')
    plt.legend()
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Weight (Standardized)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

if __name__ == "__main__":
    # Load test data
    df = pd.read_csv("/Users/lucastourinho/Downloads/heightweight/data/cleaned_height_weight_data.csv")
    X = df[['Height']]
    y = df['Weight']

    # Load the saved model
    model = load_model("/Users/lucastourinho/Downloads/heightweight/models/polynomial_model.pkl")

    # Evaluate the model
    evaluate_model(model, X, y)
