# Height-Weight Prediction Model

This project is a machine learning model that predicts human weight based on height using polynomial regression. The dataset used consists of human height and weight measurements, and the model is trained and evaluated to provide accurate predictions of weight based on height. The project includes data preprocessing, model training, model evaluation, and testing.


## Features

- **Data Preprocessing**: Handles cleaning and transformation of the dataset.
- **Polynomial Regression Model**: Predicts weight based on height with polynomial regression (degree 3).
- **Model Saving and Loading**: Trained models are saved and loaded using `joblib`.
- **Model Evaluation**: Provides key metrics like Mean Squared Error (MSE) and R-squared (R²) and includes visualizations.
- **Unit Testing**: Ensures that all components (data loading, model saving/loading, predictions) work correctly.

## Project Components
1. Data Preprocessing (data_processing.py)
Reads the original dataset (SOCR-HeightWeight.csv), cleans it, converts height from inches to centimeters and weight from pounds to kilograms, and saves the cleaned data to cleaned_height_weight_data.csv.

2. Model Training (model_training.py)
Trains a polynomial regression model (degree 3) on the cleaned data, evaluates it, and saves the trained model as polynomial_model.pkl.

3. Model Evaluation (model_evaluation.py)
Loads the saved model, evaluates its performance on test data, prints evaluation metrics (MSE and R²), and generates visualizations.

4. Utility Functions (utils.py)
Contains helper functions to load the dataset, save models, and load models.

5. Unit Tests (test_model.py)
Contains unit tests to verify data loading, model saving/loading, and model predictions.


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heightweight.git
cd heightweight
python3 -m venv venv
source venv/bin/activate  # On macOS or Linux
# For Windows use: venv\Scripts\activate
pip install -r requirements.txt
python3 src/data_processing.py
python3 src/model_training.py
python3 src/model_evaluation.py
python3 tests/test_model.py


