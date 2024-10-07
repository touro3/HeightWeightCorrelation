from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # Import joblib for saving the model

# Load the cleaned dataset
df = pd.read_csv("/Users/lucastourinho/Downloads/heightweight/data/cleaned_height_weight_data.csv")

# Define features (X) and target (y)
X = df[['Height']]  # Features (independent variable)
y = df['Weight']    # Target (dependent variable)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler and scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# Polynomial Regression (Degree 3)
# -----------------------------------------
# Create and train polynomial regression pipeline (degree 3)
poly_model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
poly_model.fit(X_train_scaled, y_train)

# Save the trained polynomial model to a pickle file
model_path = "/Users/lucastourinho/Downloads/heightweight/models/polynomial_model.pkl"
joblib.dump(poly_model, model_path)

print(f"Model saved successfully at {model_path}!")

# -----------------------------------------
# Evaluation and Visualizations (Optional)
# -----------------------------------------
# Make predictions on the test set
y_poly_pred = poly_model.predict(X_test_scaled)

# Evaluation for Polynomial Regression
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_r2 = r2_score(y_test, y_poly_pred)

print("---- Polynomial Regression (Degree 3) ----")
print(f"Mean Squared Error: {poly_mse}")
print(f"R-squared: {poly_r2}")

# Cross-Validation for Polynomial Regression
cv_scores = cross_val_score(poly_model, X_train_scaled, y_train, cv=5, scoring='r2')
print("\n---- Cross-Validation (5-fold) ----")
print(f"Cross-Validation R-squared: {cv_scores.mean()}")

# Visualize Actual vs Predicted Weights
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_poly_pred, label='Polynomial Regression (Degree 3)', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
plt.xlabel('Actual Weight (Standardized)')
plt.ylabel('Predicted Weight (Standardized)')
plt.title('Actual vs Predicted Weight')
plt.legend()
plt.show()
