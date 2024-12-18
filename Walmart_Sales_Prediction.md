# Regression Modeling: Walmart Sales Prediction
By: Zachary Raup


### Introduction:
Accurate weekly sales predictions are vital in retail for efficient inventory management, precise demand forecasting, and maximizing profitability. Sales are influenced by numerous factors, including store-specific attributes, seasonal trends, and external market conditions, making predictions a complex task. This project leverages machine learning to address this challenge, applying and comparing multiple regression models to uncover patterns and predict sales more effectively than traditional methods. By evaluating the performance of these models, the project aims to identify the most reliable approach for improving decision-making and operational efficiency across retail stores.


### Problem Statement 
The goal of this project is to predict weekly sales for retail stores using store features. Accurate predictions help businesses manage inventory, plan marketing, and allocate resources effectively. The challenge is to find a model that captures the relationships between store attributes and sales while maintaining accuracy and reliability.  

### Data Overview
The dataset, sourced from Kaggle, contains weekly sales data for various Walmart stores from 2010 to 2012, along with store-specific and economic features. The columns in the dataset are as follows:

- **Store:** The store number.
- **Date:** The week corresponding to the sales.
- **Weekly_Sales:** Total sales for the given store in a specific week.
- **Holiday_Flag:** Indicates if the week has a special holiday (1 for Yes, 0 for No).
- **Temperature**: Average temperature for the week.
- **Fuel_Price:** Cost of fuel in the region of the store.
- **CPI:** Consumer Price Index for the region.
- **Unemployment:** Prevailing unemployment rate (in percentage).

The objective is to predict Weekly_Sales based on these features. Covering two years of data across multiple stores, this dataset provides a comprehensive foundation for training regression models to forecast retail sales effectively.

#### Import Libraries for Analysis and Modeling
```python
# Import necessary libraries
import os
import warnings
warnings.filterwarnings("ignore")

# Import libraries for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing and modeling tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
```

### Retrieving the Walmart Dataset
This section retrieves the Walmart dataset from the specified file path, loads it into a DataFrame, and displays the first 10 rows for preview.
```python
# Display file paths in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load Walmart dataset
data_path = '/kaggle/input/walmart-dataset/Walmart.csv'  # Path to the dataset
df = pd.read_csv(data_path)

# Preview dataset
df.head(10)
```

### Understanding the Dataset
The following code provides an overview of the dataset, displaying its shape (number of rows and columns) along with detailed information about the columns and their data types.

```python
# Display the shape of the dataset
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Display column information and data types for clarity
print("\nDataset Information:")
df.info()
```
```python
# Display summary statistics for all numeric columns
print("\nSummary Statistics:")
df.describe()
```

### Data Cleaning
This section focuses on cleaning the dataset by performing key tasks: converting the 'Date' column to datetime format, identifying missing values, checking for duplicate rows, and detecting outliers in numeric columns. The outlier detection is achieved by calculating the Interquartile Range (IQR) and identifying values outside the typical range, helping to ensure data quality for modeling.













### Skills demonstrated in this project:

- [Data preprocessing with train_test_split and StandardScaler](#data-preprocessing-split-and-scale-data)
- [Feature Importance using RandomForest](#feature-importance-using-random-forest)
- [Hyperparameter Optimization with KNN and GridSearch](#hyperparameter-optimization-for-knn-with-gridsearch)
- [Model comparison (Logistic Regression, KNN, Decision Tree, Random Forest, SVM)](#compare-multiple-models)
- [Model evaluation using precision, recall, accuracy, and cross-validation metrics](#evaluate-models-on-test-set)
- [ROC Curves](#roc-curves)  


### About the Data:
The dataset used is sourced from [DataCamp](https://app.datacamp.com) and was a practice dataset from the [Supervised Learning with scikit-learn course]([datacamp.com/learn/courses/supervised-learning-with-scikit-learn](https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn)). It is a cleaned version of the diabetes dataset with no missing data, and the target variable, diabetes, is a binary column where 1 indicates that the patient has diabetes and 0 indicates that they do not. The "dpf" column refers to Diabetes Pedigree Function. This column provides a measure of the likelihood of diabetes based on a person's genetic history (family history) and the interaction of genetics with other risk factors. This cleaned dataset version allows this project to focus on model training, evaluation, and interpretation without worrying about data preprocessing issues like missing values.  


### Load the Cleaned Diabetes Dataset
This section imports necessary Python libraries for data manipulation and visualization. It loads the cleaned diabetes dataset into a Pandas DataFrame, providing a structured view of the data, which is essential for subsequent processing.  


```python
# Importing libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
diabetes_df = pd.read_csv('datasets/diabetes_clean.csv')

# Preview the diabetes dataset
print(diabetes_df.head())
```

```python
   pregnancies  glucose  diastolic  triceps  ...   bmi    dpf  age  diabetes
0            6      148         72       35  ...  33.6  0.627   50         1
1            1       85         66       29  ...  26.6  0.351   31         0
2            8      183         64       32  ...  23.3  0.672   32         1
3            1       89         66       23  ...  28.1  0.167   21         0
4            0      137         40       35  ...  43.1  2.288   33         1
```  

