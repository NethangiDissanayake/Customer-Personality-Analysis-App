# Customer-Personality-Analysis-App
A Streamlit-powered tool that clusters customers using KMeans/DBSCAN. From notebook to live demo in minutes!  
Predict customer segments using ML clustering
## Overview

This project aims to analyze customer data and segment customers based on their demographics and purchasing behavior. By applying unsupervised machine learning techniques—**KMeans** and **DBSCAN**—the goal is to uncover meaningful customer clusters that businesses can leverage to enhance targeted marketing and personalized services.

## Key Features

- **Comprehensive Data Preprocessing**: Cleaning, encoding, and scaling of input features to ensure high-quality inputs for clustering.
- **Feature Selection**: Careful selection of features most relevant to customer segmentation.
- **Clustering Using Two Algorithms**:
  - KMeans for finding structured, equally sized customer segments.
  - DBSCAN for identifying clusters of varying shapes and handling outliers.
- **Evaluation and Optimization**: Use of silhouette score and elbow method to evaluate clustering quality and determine optimal cluster count.
- **Cluster Interpretation**: Analysis of each cluster to understand patterns in income, product spending, and family composition.
- **Interactive Visualizations**: Scatter plots and PCA to visualize customer clusters in reduced dimensions.

## Algorithms Used

### KMeans Clustering
- A centroid-based clustering algorithm that partitions data into a predefined number of clusters (K).
- Works well for spherical, equally sized clusters.
- Optimal number of clusters determined using **Elbow Method** and **Silhouette Score**.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- A density-based clustering method that groups closely packed points and identifies outliers as noise.
- Does not require predefining the number of clusters.
- Useful for identifying clusters of arbitrary shape and dealing with anomalies in customer behavior.

## Model Performance

- **KMeans**:
  - Optimal number of clusters: *[insert K, e.g., 4]*.
  - Silhouette Score: *[insert value, e.g., 0.43]*.
  - Well-separated, interpretable clusters based on customer income and product purchases.

- **DBSCAN**:
  - Epsilon and min_samples tuned manually for best density separation.
  - Identified *[insert number]* dense clusters with *[insert number]* noise points.
  - Better at identifying outlier customers compared to KMeans.
