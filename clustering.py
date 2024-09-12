import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

# Sample DataFrame
data = {
    'Category': ['A', 'B', 'C', 'A', 'A', 'D', 'E', 'F', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
}
df = pd.DataFrame(data)

# Step 1: Define the threshold for rare categories
threshold = 2  # Minimum count for a category to be considered common

# Step 2: Group rare categories into 'Other'
category_counts = df['Category'].value_counts()
rare_categories = category_counts[category_counts < threshold].index

# Replace rare categories with 'Other'
df['Category'] = df['Category'].replace(rare_categories, 'Other')

# Step 3: One-hot encoding for categorical data
encoder = OneHotEncoder()
encoded_categories = encoder.fit_transform(df[['Category']])

# Convert the encoded categories to a DataFrame for clustering
encoded_df = pd.DataFrame(encoded_categories.toarray(), columns=encoder.get_feature_names_out(['Category']))

# Step 4: Apply K-Means clustering to group similar categories
n_clusters = 3  # Define the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(encoded_df)

# Step 5: Output the DataFrame with clusters
print("DataFrame with Clusters:")
print(df)

# Step 6: Handling outliers by calculating distances to cluster centers
def find_outliers(kmeans_model, data, threshold=2.0):
    """Identify outliers based on distance to cluster centers."""
    distances = np.min(kmeans_model.transform(data), axis=1)
    outliers = distances > threshold  # You can define a distance threshold for outliers
    return outliers

# Calculate outliers using distance to nearest cluster center
outliers = find_outliers(kmeans, encoded_df)

# Mark outliers in the DataFrame
df['Is_Outlier'] = outliers

# Step 6.1: Merging outliers with the nearest cluster 
# If an outlier is identified, you can merge it with the nearest cluster.
# You can also treat them as a separate group if needed.

df.loc[df['Is_Outlier'], 'Cluster'] = -1  # Mark outliers as a separate cluster (-1) or handle as desired

print("\nDataFrame with Outliers:")
print(df)
