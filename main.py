import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from data_prep import load_data, handle_missing_values, preprocess_data, clean_data
from eda import visualize_diabetes_distribution,analyze_correlations,plot_age_distribution,plot_glucose_vs_bmi,plot_blood_pressure_vs_age,plot_insulin_vs_skin_thickness,plot_diabetes_pedigree_function,plot_all_features_vs_outcome,plot_pie_chart,plot_violin_plot,plot_scatter_insulin_outcome,plot_pairplot
from modeling import split_data,train_model
from deployment import save_model,load_model,predict_diabetes_risk
from modelevaluation import evaluate_model, get_diabetes_risk_message

custom_css = """
    <style>
        .main {
            background-image: url("https://publichealth.gwu.edu/sites/g/files/zaxdzs4586/files/images/Diabetes%20generic.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        /* Custom styling goes here */
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
# Define the abstract content
abstract_content = """
**Abstract:**

Welcome to our Diabetes Risk Prediction tool! This web application is designed to help individuals assess their risk of developing diabetes based on various health parameters. By leveraging machine learning techniques, we analyze key factors such as glucose levels, blood pressure, BMI, insulin levels, and more to provide personalized risk assessments.

**Key Features:**

1. **Data Summary:** Explore the dataset used for training the model, including an overview of the original dataset and the preprocessed data.

2. **Exploratory Data Analysis (EDA):** Visualize and understand the distribution of diabetes risk levels, correlations between different health parameters, and explore relationships between features and diabetes outcomes through interactive plots and charts.

3. **Model Validity:** Evaluate the performance of our predictive model using metrics such as accuracy, precision, recall, and F1-score. Select your preferred features and input values to predict your diabetes risk and receive instant feedback on your predicted risk level.

4. **Suggestions:** Receive personalized suggestions based on your input values to help you manage and improve your health. Suggestions include recommendations for maintaining healthy insulin levels, managing BMI, monitoring glucose levels, and more.

5. **Visual Enhancements:** Our web application features visually appealing design elements, including custom CSS styling, interactive charts, and informative images related to diabetes.

**How to Use:**

- Navigate through the different sections using the sidebar menu.
- Select your preferred features for prediction and input your values.
- Click the "Predict" button to receive your diabetes risk prediction and personalized suggestions.
- Explore the various visualizations in the Exploratory Data Analysis section to gain insights into diabetes risk factors.


"""


def main():
    st.markdown("<h1 style='text-align: center;'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;"><img src="https://cdn.pixabay.com/photo/2019/11/03/20/33/raspberry-4599580_640.jpg" width="400"/></p>', unsafe_allow_html=True)
    # Define the file path of the dataset
    file_path = "C:/Users/Dell/KNN ML/diabetes.csv"  # Update with your file path
    load_data(file_path)
    # Sidebar with options
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select an option", ["About", "Data Summary", "Exploratory Data Analysis", "Model Validity"])
    
    # Preprocess the data
    data = preprocess_data(file_path)
    # Display About section
    if option == "About":
       st.markdown(abstract_content, unsafe_allow_html=True)

    elif option == "Data Summary":
        st.subheader("Original Dataset")
        st.write(data)
         
        # Clean the data to remove outliers
        st.subheader("Data Cleaning: Outlier Detection and Removal")
        cleaned_data = clean_data(data)
    
        # Preprocessed data
        st.subheader("Preprocessed Dataset (After Outlier Removal)")
        st.write(cleaned_data)  # Print the cleaned dataset
        st.subheader("Features Overview and Relationship with Outcome")
        st.write("The dataset contains several features that are used to predict the likelihood of an individual having diabetes (the outcome). Here's a brief overview of each feature and its potential relationship with the outcome:")

        st.write("- **Glucose:** Glucose levels in the blood are a key indicator of diabetes risk. Higher glucose levels are associated with an increased risk of diabetes.")

        st.write("- **Blood Pressure:** Elevated blood pressure levels may indicate an increased risk of diabetes and other health complications.")

        st.write("- **Skin Thickness:** While skin thickness itself may not directly cause diabetes, abnormal skin thickness measurements could be a symptom of underlying health issues related to diabetes.")

        st.write("- **Insulin:** Insulin is a hormone that regulates blood sugar levels. Abnormal insulin levels can indicate insulin resistance, a common precursor to type 2 diabetes.")

        st.write("- **BMI (Body Mass Index):** Higher BMI values are often associated with obesity, which is a major risk factor for type 2 diabetes.")
  
        st.write("- **Diabetes Pedigree Function:** This function provides information about the likelihood of diabetes based on family history. A higher value indicates a stronger family history of diabetes, which can increase an individual's risk.")

        st.write("- **Age:** Age is a significant risk factor for diabetes, with the risk increasing as individuals get older.")

        st.write("By analyzing these features in combination, machine learning models can make predictions about an individual's diabetes risk.")

    elif option == "Exploratory Data Analysis":
        # Explore the data
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Exploratory Data Analysis")
        # Visualize the distribution of diabetes risk levels
        st.subheader("Distribution of Diabetes Risk Levels")
        visualize_diabetes_distribution(data)

        # Analyze correlations between features and diabetes risk
        st.subheader("Correlation Matrix")
        analyze_correlations(data)

        # Pie chart for distribution of diabetes outcomes
        st.subheader("Distribution of Diabetes Outcomes")
        plot_pie_chart(data, 'Outcome')

        # Violin plot for Age vs Outcome
        st.subheader("Violin Plot: Age vs Outcome")
        plot_violin_plot(data, 'Age', 'Outcome')

        # Scatter plot for Insulin vs Outcome
        st.subheader("Scatter Plot: Insulin vs Outcome")
        plot_scatter_insulin_outcome(data)

        # Pair plot for pairwise relationships
        st.subheader("Pairwise Relationships")
        plot_pairplot(data)

        # Histogram for Age distribution
        st.subheader("Age Distribution")
        plot_age_distribution(data)

        # Scatter plot for Glucose vs BMI
        st.subheader("Glucose vs BMI")
        plot_glucose_vs_bmi(data)

        # Scatter plot for Blood Pressure vs Age
        st.subheader("Blood Pressure vs Age")
        plot_blood_pressure_vs_age(data)

        # Scatter plot for Insulin vs Skin Thickness
        st.subheader("Insulin vs Skin Thickness")
        plot_insulin_vs_skin_thickness(data)

        # Histogram for Diabetes Pedigree Function distribution
        st.subheader("Diabetes Pedigree Function Distribution")
        plot_diabetes_pedigree_function(data)

        # Box plots for all features vs Outcome
        st.subheader("Box Plots: Features vs Outcome")
        plot_all_features_vs_outcome(data)


    elif option == "Model Validity":
        # Feature selection
        selected_features = st.sidebar.multiselect("Select features", data.columns)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(data[selected_features + ['Outcome']])

        # Train model
        model = train_model(X_train, y_train)

       
        # Evaluate model
        accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-score: {f1}")

        # Print classification report
        st.subheader("Classification Report")
        st.code(report)

        # Explain features in the classification report
        st.subheader("Understanding the Classification Report:")

        # Precision
        st.write("- Precision: Precision tells us how many of the predicted diabetic cases are actually diabetic. So, higher precision means fewer false alarms.")

        # Recall
        st.write("- Recall: Recall tells us how many of the actual diabetic cases were correctly predicted by the model. Higher recall means fewer missed diabetic cases.")

        # F1-score
        st.write("- F1-score: F1-score is a balance between precision and recall. A higher F1-score means the model is good at both avoiding false alarms and not missing actual diabetic cases.")
 
        # Support
        st.write("- Support: Support shows how many actual diabetic and non-diabetic cases were in our test dataset. It helps us understand the context of precision, recall, and F1-score.")

        st.write("- **Macro Average**: The macro average provides an overall assessment of the model's performance across all classes, giving equal weight to each class. A high macro average indicates good performance across all classes, irrespective of their sizes. However, it may not accurately reflect performance in imbalanced datasets.")

        st.write("- **Weighted Average**: The weighted average considers the class distribution, giving more weight to classes with more instances. It provides a more reliable measure of performance, especially in imbalanced datasets. A high weighted average suggests that the model performs well overall, considering the distribution of classes in the dataset.")

        # Change features to predict diabetes
        st.title("Predict Diabetes")
        st.write("Enter values for the following features (general range):")
        st.sidebar.write("Range of the features")
        st.sidebar.write("- Glucose (70-200)")
        st.sidebar.write("- Blood Pressure (50-110)")
        st.sidebar.write("- Skin Thickness (10-100)")
        st.sidebar.write("- Insulin (0-600)")
        st.sidebar.write("- BMI (10-50)")
        st.sidebar.write("- Diabetes Pedigree Function (0.0-2.0)")
        st.sidebar.write("- Age (20-90)")

        feature_values = {}
        for feature in selected_features:
            feature_values[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=1000.0)
        
        if st.button("Predict"):
            input_data = pd.DataFrame([feature_values])
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.markdown("<h3>Prediction: <span style='color:red'>Diabetic</span></h3>", unsafe_allow_html=True)

                st.write("Based on the provided features, it is predicted that the individual is at risk of diabetes. We recommend consulting a healthcare professional for further evaluation and advice.")
            else:
                st.markdown("<h3>Prediction: <span style='color:green'>Non-Diabetic</span></h3>", unsafe_allow_html=True)
                st.write("Based on the provided features, it is predicted that the individual is not at risk of diabetes. However, maintaining a healthy lifestyle is always beneficial.")
        # Suggestions based on feature levels
        if 'Insulin' in feature_values and feature_values['Insulin'] > 300:
            st.write("- Your insulin level is quite high. Please consult a doctor for a proper diet plan.")
        if 'BMI' in feature_values and feature_values['BMI'] > 30:
            st.write("- Your BMI indicates that you are overweight. Consider maintaining a healthy diet and regular exercise routine.")
        if 'Glucose' in feature_values and feature_values['Glucose'] > 150:
            st.write("- Your blood glucose level is elevated. It's essential to monitor your sugar intake and consult a healthcare provider.")
        if 'BloodPressure' in feature_values and feature_values['BloodPressure'] > 130:
            st.write("- Your blood pressure is high. It's important to monitor it regularly and follow your doctor's recommendations for managing hypertension.")
        # Display images based on prediction
        if 'prediction' in locals():
            if prediction == 1:
               st.markdown('<p style="text-align:center;"><img src="https://onlinepublichealth.gwu.edu/wp-content/uploads/sites/47/2021/03/Badge-1.png" alt="Diabetic" width="300"/></p>', unsafe_allow_html=True)
            elif prediction == 0:
               st.markdown('<p style="text-align:center;"><img src="https://i.pinimg.com/736x/96/56/6d/96566d3870b4544b9b033e9cbaff7b25.jpg" alt="Non-Diabetic" width="300"/></p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
