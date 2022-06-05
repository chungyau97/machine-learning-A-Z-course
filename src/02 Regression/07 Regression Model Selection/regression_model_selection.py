# Model Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def print_result(model, y_test, y_pred):
    np.set_printoptions(precision=2)
    print(model)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    print(r2_score(y_test, y_pred)) 

# ================ Multiple Linear Regression model ================
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lin_reg.predict(X_test)

# Evaluating the Model Performance
print_result("Multiple Linear Regression model", y_test, y_pred)
# ================================ END ================================

# ==================== Polynomial Regression model ==================== 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg1 = PolynomialFeatures(degree = 4)
X_poly = poly_reg1.fit_transform(X_train)
poly_reg2 = LinearRegression()
poly_reg2.fit(X_poly, y_train)

# Predicting the Test set results
y_pred = poly_reg2.predict(poly_reg1.transform(X_test))

# Evaluating the Model Performance
print_result("Polynomial Regression model", y_test, y_pred)
# ================================ END ================================

# ============================= SVR model ============================= 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_fs_train = sc_X.fit_transform(X_train)
y_fs_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(X_fs_train, np.ravel(y_fs_train))

# Predicting the Test set results
y_pred = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(X_test)).reshape(len(X_test),1))

# Evaluating the Model Performance
print_result("SVR model", y_test, y_pred)
# ================================ END ================================ 

# =================== Decision Tree Regression model ================== 
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state = 0)
dt_reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dt_reg.predict(X_test)

# Evaluating the Model Performance
print_result("Decision Tree Regression model", y_test, y_pred)
# ================================ END ================================

# ================== Random Forest Regression model =================== 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, np.ravel(y_train))

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Evaluating the Model Performance
print_result("Random Forest Regression model", y_test, y_pred)
# ================================ END ================================