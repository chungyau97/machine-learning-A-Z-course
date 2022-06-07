# Logistic Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

# Training the Logistic Regression model on the Training set

# Predicting the Test set results

# Making the Confusion Matrix

# Computing the accuracy with k-Fold Cross Validation