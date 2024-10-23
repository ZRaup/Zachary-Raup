# Predicting Diabetes Using Machine Learning | Comparison of Classification Models 
By: Zachary Raup

## Project Introduction:
This project demonstrates how to classify whether a patient has diabetes using various machine learning models and techniques within the scikit-learn library. With diabetes being a leading cause of serious health issues, including heart disease, kidney failure, and vision loss, early detection and intervention are crucial for effective management and treatment. By leveraging machine learning algorithms, we can enhance diagnostic accuracy, enabling healthcare providers to identify at-risk individuals more efficiently.

The models applied in this project include Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM). To enhance model performance, techniques such as scaling, Random Forest feature importance analysis, and hyperparameter optimization are utilized. The models are evaluated using cross-validation metrics and visualized through confusion matrices and ROC curves, allowing for a comprehensive assessment of their effectiveness.

Ultimately, this project aims to provide insights into the best-performing models for diabetes prediction, contributing to more accurate and timely healthcare decisions. The findings may inform future research and applications in predictive analytics within the healthcare domain, potentially leading to better patient outcomes and reduced healthcare costs.


### Skills demonstrated in this project:

- [Data preprocessing with train_test_split and StandardScaler](#data_preprocessing:_split_and_scale_data)
- [Feature importance using Random Forest](#feature_importance_using_random_forest)
- [Hyperparameter Optimization with KNN and GridSearch](#hyperparameter_optimization_for_knn_with_gridsearch)
- [Model comparison (Logistic Regression, KNN, Decision Tree, Random Forest, SVM)](#compare_multiple_models)
- [Model evaluation using precision, recall, accuracy, and cross-validation metrics](#evaluate_models_on_test_set)
- [ROC Curves](#roc_curves)


### About the Data:
The dataset used is sourced from [DataCamp](https://app.datacamp.com) and was a practice dataset from the [Supervised Learning with scikit-learn course]([datacamp.com/learn/courses/supervised-learning-with-scikit-learn](https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn)). It is a cleaned version of the diabetes dataset with no missing data, and the target variable, diabetes, is a binary column where 1 indicates that the patient has diabetes and 0 indicates that they do not. The "dpf" column refers to Diabetes Pedigree Function. This column provides a measure of the likelihood of diabetes based on a person's genetic history (family history) and the interaction of genetics with other risk factors. This cleaned dataset version allows this project to focus on model training, evaluation, and interpretation without worrying about data preprocessing issues like missing values.


### Load the Cleaned Diabetes Dataset
This section imports necessary Python libraries for data manipulation and visualization. It loads the cleaned diabetes dataset into a Pandas DataFrame, providing a structured view of the data, which is essential for subsequent processing.

# Importing libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
diabetes_df = pd.read_csv('datasets/diabetes_clean.csv')

# Preview the diabetes dataset
print(diabetes_df.head())


   pregnancies  glucose  diastolic  triceps  ...   bmi    dpf  age  diabetes
0            6      148         72       35  ...  33.6  0.627   50         1
1            1       85         66       29  ...  26.6  0.351   31         0
2            8      183         64        0  ...  23.3  0.672   32         1
3            1       89         66       23  ...  28.1  0.167   21         0
4            0      137         40       35  ...  43.1  2.288   33  


### Data Preprocessing: Split and Scale Data
Assign the feature variables (X) and target variable (y) and randomly split the dataset into training (80%) and testing (20%) sets with train_test_split from sklearn. The stratification ensures that the class distribution of the target variable (diabetes/no diabetes) is maintained, which is critical for balanced model evaluation. This section also applies standard scaling to the data. Scaling ensures that all feature values are normalized and on the same scale, which assists models like KNN and SVM that are sensitive to differences in feature magnitudes.
