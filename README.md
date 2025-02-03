# CMSE802 Project: Prediction of Dislocation Mobility in TaW Alloy

## Project Description

This project aims to use machine learning to predict the dislocation mobility in Tantalum-Tungsten (TaW) alloy, focusing on how solute atoms affect dislocation motion. Instead of using molecular dynamics simulations, I will use existing datasets and machine learning algorithms like XGBoost from a published scientific paper to estimate dislocation velocity based on parameters like solute concentration, temperature, and applied stress.

## Project Objectives

- **Data Collection & Preprocessing:** Load and clean the dataset.
- **Model Development:** Train an XGBoost regression model to predict dislocation velocity.
- **Model Evaluation:** Assess model performance using RMSE and visualizations.
- **Data Visualization:** Plot results comparing dislocation velocity against parameters like stress, temperature, and W content.

## Folder Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Contains Jupyter notebooks for exploratory analysis.
- `code/`: Any written code for preprocessing, model building, and utility functions.
- `results/`: Any result will be stored here, such as plots.
- `python_libraries.txt`: Lists of Python libraries required to run the code I'm using.
