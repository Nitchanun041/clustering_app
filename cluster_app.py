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

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset by Nitchanun Benjawan")

# Sidebar slider
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

# K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting the clusters
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Legend
for i in range(k):
    ax.scatter([], [], label=f"Cluster {i}", color=scatter.cmap(scatter.norm(i)))
ax.legend()

# Show plot
st.pyplot(fig)

