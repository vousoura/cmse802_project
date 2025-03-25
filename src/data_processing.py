import pandas as pd
import numpy as np
# Load the raw dataset
def load_data(filepath="../data/raw/raw_data.xlsx"):  # No spaces in filename
    """Load the dataset from an Excel file."""
    return pd.read_excel(filepath)

# Process data
def process_data(df):
    """Clean the dataset by dropping missing values and filtering slip planes."""
    df = df.dropna()  # Remove missing values
    
    # Define relevant slip planes
    slip_planes_of_interest = [110, 112, 123]

    # Filter dataset to include only the selected slip planes
    df = df[df["Slip_plane"].isin(slip_planes_of_interest)]

    # Rename "W" column to "Solute_Concentration"
    df = df.rename(columns={"W": "Solute_Concentration"})

    return df

# Main function to execute cleaning and saving
def main():
    # Load the dataset
    df = load_data()

    # Process the data
    filtered_data = process_data(df)

    # Save cleaned data as CSV
    filtered_data.to_csv("../data/processed/cleaned_data.csv", index=False)

    print("Cleaned data saved successfully!")

# Run script
if __name__ == "__main__":
    main()

