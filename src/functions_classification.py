import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
import umap 
import hdbscan


df = pd.read_csv('../data/AppML_ZTF_table.csv')

# Drop identifier/metadata columns
drop_cols = ["id", "objectId", 'decstd', 'rastd', "htm16", "ssnamenr", "tns_name", "tns_type",
             "host_name", "classification", "classificationReliability"]
features = df.drop(columns=drop_cols)

# Check missingness — crucial before choosing imputation strategy
print(features.isnull().mean().sort_values(ascending=False).head(20))

# Impute (median is robust to outlier-heavy astro data)
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(features)

# Scale — RobustScaler handles the heavy-tailed distributions typical in ZTF
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# PCA to check variance explained
pca = PCA().fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
# Find n_components for ~90% variance

# UMAP for visualization and as clustering input
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
embedding = reducer.fit_transform(X_scaled)

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean")
labels = clusterer.fit_predict(embedding)  # or X_scaled directly

# -1 = noise points in HDBSCAN
print(pd.Series(labels).value_counts())

from sklearn.metrics import adjusted_rand_score, silhouette_score

# External validation against known labels
known_mask = df["classification"].notna()
ari = adjusted_rand_score(df.loc[known_mask, "classification"], labels[known_mask])

# Internal validation
sil = silhouette_score(X_scaled, labels)

# make a plot of current clusters in UMAP space
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels)
plt.title(f"HDBSCAN Clusters in UMAP Space\nARI: {ari:.2f}, Silhouette: {sil:.2f}")
plt.legend(title="Cluster")
plt.show()

# plot the outliers and show their classifications according to the TNS catalog
outlier_scores = clusterer.outlier_scores_
print("Outlier scores for noise points (label=-1):", outlier_scores[labels == -1])
plt.hist(outlier_scores, bins=50)
plt.title("HDBSCAN Outlier Scores")
plt.xlabel("Outlier Score")
plt.ylabel("Frequency")
plt.show()