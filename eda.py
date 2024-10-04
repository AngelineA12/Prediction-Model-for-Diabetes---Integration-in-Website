# eda.py

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def visualize_diabetes_distribution(data):
    """
    Visualize the distribution of diabetes risk levels.
    """
    sns.countplot(x='Outcome', data=data)
    plt.title('Distribution of Diabetes Risk Levels')
    plt.xlabel('Diabetes Risk Level')
    plt.ylabel('Count')
    st.pyplot()

def analyze_correlations(data):
    """
    Analyze correlations between features and diabetes risk.
    """
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot()

def plot_age_distribution(data):
    """
    Visualize the distribution of ages in the dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot()

def plot_glucose_vs_bmi(data):
    """
    Visualize the relationship between Glucose and BMI.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Glucose', y='BMI', data=data, hue='Outcome')
    plt.title('Glucose vs BMI')
    plt.xlabel('Glucose')
    plt.ylabel('BMI')
    st.pyplot()

def plot_blood_pressure_vs_age(data):
    """
    Visualize the relationship between Blood Pressure and Age.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='BloodPressure', y='Age', data=data, hue='Outcome')
    plt.title('Blood Pressure vs Age')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Age')
    st.pyplot()

def plot_insulin_vs_skin_thickness(data):
    """
    Visualize the relationship between Insulin and Skin Thickness.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Insulin', y='SkinThickness', data=data, hue='Outcome')
    plt.title('Insulin vs Skin Thickness')
    plt.xlabel('Insulin')
    plt.ylabel('Skin Thickness')
    st.pyplot()

def plot_diabetes_pedigree_function(data):
    """
    Visualize the distribution of Diabetes Pedigree Function.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['DiabetesPedigreeFunction'], bins=20, kde=True)
    plt.title('Diabetes Pedigree Function Distribution')
    plt.xlabel('Diabetes Pedigree Function')
    plt.ylabel('Count')
    st.pyplot()

def plot_all_features_vs_outcome(data):
    """
    Visualize the relationship between all features and the outcome.
    """
    numerical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    plt.figure(figsize=(15, 10))  # Adjust figure size
    for i, feature in enumerate(numerical_features, start=1):
        plt.subplot(3, 3, i)  # Create subplots
        sns.boxplot(x='Outcome', y=feature, data=data)
        plt.title(f'{feature} vs Outcome')
        plt.xlabel('Outcome')
        plt.ylabel(feature)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    st.pyplot()

def plot_pie_chart(data, column):
    """
    Visualize the distribution of categories in a categorical column using a pie chart.
    """
    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.3))
    plt.title(f'{column} Distribution')
    plt.ylabel('')
    plt.tight_layout()  # Add this line to create a tight layout
    st.pyplot()

def plot_violin_plot(data, x, y):
    """
    Visualize the distribution of a numerical variable across different categories using a violin plot.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=x, y=y, data=data, palette='pastel', inner='quartile')
    plt.title(f'Violin Plot: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()  # Add this line to create a tight layout
    st.pyplot()

def plot_scatter_insulin_outcome(data):
    """
    Visualize the relationship between 'Insulin' and 'Outcome' using a scatter plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Insulin', y='Outcome', data=data, hue='Outcome', palette='pastel')
    plt.title('Scatter Plot: Insulin vs Outcome')
    plt.xlabel('Insulin')
    plt.ylabel('Outcome')
    plt.yticks([0, 1], ['Non-Diabetic', 'Diabetic'])
    st.pyplot()

def plot_pairplot(data):
    """
    Visualize pairwise relationships in the dataset using a pair plot.
    """
    sns.pairplot(data, hue='Outcome', palette='pastel')
    st.pyplot()
