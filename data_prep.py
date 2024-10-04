# data_prep.py
import streamlit as st

import pandas as pd

def load_data(file_path):
    """
    Load patient data from CSV file.
    """
    return pd.read_csv(file_path)

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    """
    return data.dropna()

def remove_outliers(data, column):
    """
    Remove outliers from a numerical column in the dataset.
    """
    # Add your code for outlier removal here
    # Example: Remove outliers using IQR method
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def clean_data(data):
    """
    Clean the dataset by removing outliers.
    """
    # Example: Remove outliers from numerical columns
    numerical_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    total_outliers = 0
    cleaned_data = data.copy()  # Make a copy of the original data
    removed_data = pd.DataFrame(columns=data.columns)  # Initialize DataFrame to store removed outliers

    original_rows = cleaned_data.shape[0]  # Number of rows before cleaning

    for column in numerical_columns:
        # Count outliers for each feature
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = cleaned_data[(cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)]
        outliers_count = outliers.shape[0]
        total_outliers += outliers_count
        st.write(f"{outliers_count} outliers detected in {column}.")
        # Store removed outliers
        removed_data = removed_data.append(outliers)
        # Remove outliers from cleaned data
        cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]
    
    cleaned_rows = cleaned_data.shape[0]  # Number of rows after cleaning

    if total_outliers > 0:
        st.write(f"Total outliers detected in the dataset: {total_outliers}")
        st.write("Outliers removed successfully using the interquartile range method.")
    else:
        st.write("No outliers were detected in the dataset.")

    st.write(f"Number of rows before cleaning: {original_rows}")
    st.write(f"Number of rows after cleaning: {cleaned_rows}")

    # Display removed dataset
    if not removed_data.empty:
        st.subheader("Removed Outliers")
        st.write(removed_data)

    return cleaned_data





def preprocess_data(file_path):
    """
    Preprocess the data by handling missing values, cleaning, and encoding categorical variables.
    """
    data = load_data(file_path)
    
    
   
    return data