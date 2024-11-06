# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %%
df = pd.read_csv("sales_data_sample.csv", encoding="unicode_escape")

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.drop(['ADDRESSLINE2', 'STATE', 'POSTALCODE', 'TERRITORY'], axis=1, inplace=True)

# %%
df.isnull().sum()

# %%
X = df[['SALES', 'QUANTITYORDERED', 'PRICEEACH']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
inertia = []
K = range(1,11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# %%
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'Cluster']])

# %%
# Visualize the clusters (for example, with scatter plots)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['QUANTITYORDERED', 'SALES']])
df['Cluster'] = kmeans.fit_predict(scaled_data)
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(df[df['Cluster'] == cluster]['QUANTITYORDERED'], df[df['Cluster'] == cluster]['SALES'], label=f'Cluster {cluster}')
plt.title(f'K-Means Clustering with {optimal_k} Clusters')
plt.xlabel('QUANTITYORDERED')
plt.ylabel('SALES')
plt.legend()
plt.show()


