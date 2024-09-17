import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate Dummy Data for Regression
# 100 samples, 10 features, only 5 features are informative
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=42)

# Convert the data into a DataFrame for easier understanding
df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df['Target'] = y

print("Sample of the dataset:\n", df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Lasso Regression for Feature Selection
# Initialize Lasso model with alpha (regularization strength)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Get coefficients of the Lasso model
lasso_coefficients = lasso.coef_

# Print out the feature importance (non-zero coefficients are selected features)
print("\nLasso Regression Coefficients:")
for i, coef in enumerate(lasso_coefficients):
    print(f"Feature_{i+1}: {coef}")

# 2. Recursive Feature Elimination (RFE) for Feature Selection
# Initialize RFE with a linear model (e.g., Lasso or any other)
rfe = RFE(estimator=Lasso(alpha=0.1), n_features_to_select=5)
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = rfe.support_
ranking = rfe.ranking_

print("\nRFE Feature Selection Results:")
for i, selected in enumerate(selected_features):
    print(f"Feature_{i+1}: {'Selected' if selected else 'Not Selected'}, Ranking: {ranking[i]}")

# Testing performance on test data using selected features (from RFE or Lasso)
X_train_rfe = X_train[:, selected_features]
X_test_rfe = X_test[:, selected_features]

lasso_rfe_model = Lasso(alpha=0.1)
lasso_rfe_model.fit(X_train_rfe, y_train)
y_pred = lasso_rfe_model.predict(X_test_rfe)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Performance (MSE) with Selected Features: {mse}")
