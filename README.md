Employee Performance and Retention Analysis
Project Overview
This project provides a comprehensive analysis of an employee dataset to identify key factors influencing performance and attrition (employee turnover). It follows a structured approach, from initial data cleaning and exploration to building predictive models using both traditional machine learning and deep learning techniques. The final phase focuses on interpreting the results and providing actionable business recommendations.
The analysis is conducted using simple, core Python libraries to ensure the code is robust, easy to understand, and free from complex dependency-related errors.

What's Included
The Python script employee_analysis.py contains the complete workflow, broken down into the following phases:

Phase 1: Data Collection & EDA
Data Preprocessing: Handles missing values, removes duplicate entries, and cleans data to ensure consistency.
Exploratory Data Analysis (EDA): Generates descriptive statistics and visualizations (scatter plots, heatmaps, boxplots) to understand data characteristics and relationships.

Phase 2: Predictive Modeling
Feature Engineering: Prepares the data for modeling by encoding categorical features and scaling numerical ones.
Attrition Prediction: Builds a Logistic Regression model to predict whether an employee is likely to leave the company.
Performance Prediction: Creates a Linear Regression model to predict an employee's performance score based on other factors.

Phase 3: Deep Learning Models
Neural Networks: Implements simple, two-layer neural networks using TensorFlow/Keras for both the performance and attrition prediction tasks, showcasing a more advanced modeling approach.

Phase 4: Reporting & Insights
Key Findings: Summarizes the most significant discoveries from the analysis, such as the strong correlation between performance and attrition.
Actionable Recommendations: Provides business-oriented advice based on the data, such as developing targeted retention programs and using predictive models for early intervention.

How to Run the Code
Prerequisites
Before running the code, ensure you have the following libraries installed. You can install them using pip:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy

Files
employee_analysis.py: The main Python script containing all the code.
Employee_retention_performance.csv: The dataset used for the analysis. You need to have this file in the same directory as the script.

Execution
To run the analysis, simply execute the Python script from your terminal:
python employee_analysis.py
The script will print the analysis results to the console and display the generated plots.

Key Insights from the Analysis
Performance is a Major Indicator: A clear relationship exists between an employee's performance score and their likelihood of leaving the company.
Departmental Differences: The analysis reveals statistically significant differences in average performance scores across various departments, suggesting that a one-size-fits-all approach to talent management may be ineffective.
Predictive Power: The models demonstrate that the provided employee data is highly predictive of both performance and attrition, enabling data-driven decision-making for retention strategies.
