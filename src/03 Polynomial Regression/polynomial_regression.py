# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset

# Training the Polynomial Regression model on the whole dataset

# Visualising the Linear Regression results

# Visualising the Polynomial Regression results

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# Predicting a new result with Linear Regression

# Predicting a new result with Polynomial Regression