import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Not really needed for Least Squares, but keeping it in case

# Load the dataset
def load_data(filepath="../data/processed/cleaned_data.csv"):
    """Load the cleaned dataset from a CSV file."""
    return pd.read_csv(filepath)

# Train and evaluate Least Squares Regression
def least_squares_regression(X, y):
    """Perform Least Squares Regression using statsmodels OLS."""
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()
    return model

# Main function
def main():
    # Load dataset
    df = load_data()

    # Define features (X) and target variable (y)
    X = df[["Solute_Concentration", "Slip_plane", "Temperature", "Applied_stress"]]
    y = df["Velocity"]

    # Perform Least Squares Regression
    ls_model = least_squares_regression(X, y)

    # Print model summary
    print("\nLeast Squares Regression Summary:\n")
    print(ls_model.summary())

    # Predictions
    y_pred = ls_model.predict(sm.add_constant(X))

    # Compute RMSE
    rmse_ls = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"Least Squares RMSE: {rmse_ls}")

    # Save RMSE for later comparison
    os.makedirs("../results", exist_ok=True)  # Ensure directory exists
    with open("../results/least_squares_rmse.txt", "w") as f:
        f.write(str(rmse_ls))

    # Scatter plot: Actual vs Predicted Values
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y, y=y_pred, color="blue")
    plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='red')  # Ideal fit line
    plt.xlabel("Actual Velocity (m/s)")
    plt.ylabel("Predicted Velocity (m/s)")
    plt.title("Least Squares: Actual vs Predicted")
    plt.grid(True)
    plt.show()

# Run script
if __name__ == "__main__":
    main()
