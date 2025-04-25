# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# clustering_app.py

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# App title
st.title("🔍 K-Means Clustering App with Iris Dataset by Nitchanun Benjawan")

# Sidebar
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Create legend
legend_labels = [f"Cluster {i}" for i in range(k)]
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=scatter.cmap(scatter.norm(i)),
                      label=label, markersize=10)
           for i, label in enumerate(legend_labels)]
ax.legend(handles=handles)

# Show plot
st.pyplot(fig)


