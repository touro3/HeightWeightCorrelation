import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)
    return missing_values

def preprocess_data(df):
    """Clean and preprocess the data by converting height and weight to metric units."""
    print("Columns in the dataset:", df.columns)  # Check the column names
    # Convert height from inches to centimeters
    df['Height'] = df['Height'] * 2.54
    # Convert weight from pounds to kilograms
    df['Weight'] = df['Weight'] * 0.453592
    return df.dropna()

def remove_outliers(df, column):
    """Remove outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_cleaned

def scale_features(df):
    """Standardize the height and weight features."""
    scaler = StandardScaler()
    df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
    return df

def save_cleaned_data(df, output_filepath):
    """Save the cleaned data to a CSV file."""
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned data saved to {output_filepath}")

if __name__ == "__main__":
    # Load the dataset
    data = load_data("/Users/lucastourinho/Downloads/heightweight/data/SOCR-HeightWeight.csv")
    
    # Check for missing values
    check_missing_values(data)

    # Preprocess the data (convert units)
    processed_data = preprocess_data(data)

    # Remove outliers
    processed_data = remove_outliers(processed_data, 'Height')
    processed_data = remove_outliers(processed_data, 'Weight')

    # Scale height and weight features
    processed_data = scale_features(processed_data)

    # Save the cleaned data
    save_cleaned_data(processed_data, "/Users/lucastourinho/Downloads/heightweight/data/cleaned_height_weight_data.csv")

    print("Preprocessing complete!")
