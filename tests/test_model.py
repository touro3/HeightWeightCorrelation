import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils import load_data, save_model, load_model


class TestModel(unittest.TestCase):

    def test_load_data(self):
        """Test if the data loads correctly."""
        df = load_data("/Users/lucastourinho/Downloads/heightweight/data/cleaned_height_weight_data.csv")
        # Test if the dataframe has the correct shape (should have at least 2 columns: Height and Weight)
        self.assertEqual(df.shape[1], 3)  # Assuming columns: Index, Height, Weight
        self.assertIn('Height', df.columns)
        self.assertIn('Weight', df.columns)

    def test_model_save_and_load(self):
        """Test if the model can be saved and loaded properly."""
        # Create a simple linear regression model
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())

        # Fit the model on some sample data
        X = np.array([[1.5], [2.0], [2.5], [3.0]])
        y = np.array([3, 4, 5, 6])
        model.fit(X, y)

        # Save the model
        save_model(model, "/tmp/test_model.pkl")

        # Load the model back
        loaded_model = load_model("/tmp/test_model.pkl")

        # Check if the loaded model is the same type as the original model
        self.assertIsInstance(loaded_model, type(model))

    def test_model_prediction(self):
        """Test if the model can make predictions correctly."""
        # Create a simple polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())

        # Fit the model on some sample data
        X = np.array([[1.5], [2.0], [2.5], [3.0]])
        y = np.array([3, 4, 5, 6])
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(np.array([[2.0]]))

        # Check if predictions are reasonable (should be close to 4.0 for this test data)
        np.testing.assert_almost_equal(predictions, [4.0], decimal=1)

    def test_model_on_real_data(self):
        """Test model performance on the real dataset."""
        # Load the real dataset
        df = load_data("/Users/lucastourinho/Downloads/heightweight/data/cleaned_height_weight_data.csv")
        X = df[['Height']].values
        y = df['Weight'].values

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Check model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Test if the model's performance metrics are within expected ranges
        self.assertLess(mse, 1.0)  # MSE should be less than 1.0 (based on your previous results)
        self.assertGreater(r2, 0.2)  # R-squared should be greater than 0.2

if __name__ == "__main__":
    unittest.main()
