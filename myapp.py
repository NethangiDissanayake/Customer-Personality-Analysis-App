import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

st.title("Customer Personality Analysis")

st.markdown("Upload your dataset to perform clustering.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    df_clean = df.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean.select_dtypes(include=['int64', 'float64']))

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    n_clusters = st.slider("Select number of KMeans clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap="viridis")
    ax.set_title("KMeans Clusters")
    st.pyplot(fig)
