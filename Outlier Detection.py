import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Create dummy dataset
np.random.seed(42)
# Generate inliers
inliers = np.random.normal(0, 1, (100, 2))
# Generate outliers
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
# Combine inliers and outliers into one dataset
data = np.vstack([inliers, outliers])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])

# Scale data for DBSCAN and One-Class SVM
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
df['IsolationForest'] = iso_forest.fit_predict(df[['Feature 1', 'Feature 2']])

# One-Class SVM
one_class_svm = OneClassSVM(gamma='auto', nu=0.2)
df['OneClassSVM'] = one_class_svm.fit_predict(data_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
df['DBSCAN'] = dbscan.fit_predict(data_scaled)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

methods = ['IsolationForest', 'OneClassSVM', 'DBSCAN']
for i, method in enumerate(methods):
    axes[i].scatter(df['Feature 1'], df['Feature 2'], c=df[method], cmap='coolwarm')
    axes[i].set_title(f"Outliers Detected by {method}")
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Print results
df.head()
