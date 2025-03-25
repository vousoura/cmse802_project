import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(filepath="../data/processed/cleaned_data.csv"):
    """Load the cleaned dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

# Train and evaluate Random Forest Regression
def random_forest_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest Regression model."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Compute RMSE manually

    return rf, rmse, y_pred

# Main function
def main():
    # Load dataset
    df = load_data()

    # Define features (X) and target variable (y)
    X = df[["Solute_Concentration", "Slip_plane", "Temperature", "Applied_stress"]]
    y = df["Velocity"]

    # Split dataset into training (75%) and testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Random Forest model
    rf_model, rf_rmse, y_pred = random_forest_regression(X_train, X_test, y_train, y_test)

    # Ensure `models/` and `results/` directories exist
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # Save trained model
    model_path = "../models/random_forest_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"Model saved at {model_path}")

    # Save RMSE for later comparison
    with open("../results/random_forest_rmse.txt", "w") as f:
        f.write(str(rf_rmse))

    # Print RMSE
    print(f"\nRandom Forest RMSE: {rf_rmse}")

    # Scatter plot: Actual vs Predicted Values
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Ideal fit line
    plt.xlabel("Actual Velocity (m/s)")
    plt.ylabel("Predicted Velocity (m/s)")
    plt.title("Random Forest: Actual vs Predicted")
    plt.grid(True)
    plt.show()

# Run script
if __name__ == "__main__":
    main()
