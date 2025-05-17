import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Title and description
st.set_page_config(page_title="Customer Personality Analysis", layout="wide")
st.title("🧠 Customer Personality Analysis App")
st.markdown("Upload your customer dataset below to perform KMeans clustering and visualize personality segments.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Data Preprocessing
    df_clean = df.dropna()
    num_df = df_clean.select_dtypes(include=['int64', 'float64'])

    if num_df.empty:
        st.error("No numerical features found for clustering.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(num_df)

        # PCA for 2D Visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        # KMeans clustering
        st.sidebar.header("🛠️ Clustering Settings")
        n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 4)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        df_clean["Cluster"] = labels

        # Visualization
        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap="viridis", s=50)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("🌀 KMeans Clustering Result")

        st.subheader("📊 Cluster Visualization")
        st.pyplot(fig)

        # Display predicted clusters
        st.subheader("📌 Clustered Data with Predictions")
        st.dataframe(df_clean.assign(Cluster=labels))
