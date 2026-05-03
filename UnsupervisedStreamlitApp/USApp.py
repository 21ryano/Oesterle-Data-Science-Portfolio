# =========================================
# Unsupervised Machine Learning Streamlit App
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import datasets

from scipy.cluster.hierarchy import dendrogram, linkage


# Set the background color to light Purple
st.markdown(
    """
    <style>
    /* Change background color of the entire app */
    .stApp {
        background-color: #FAFDFF;  /* Light pastel Purple */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Change sidebar background to pastel purple
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #F0F8FF;  /* Pastel purple */
    }

    /* Optional: change sidebar text color for better contrast */
    [data-testid="stSidebar"] .css-1d391kg {
        color: #000000;  /* Black text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Change top menu bar color to Purple
st.markdown(
    """
    <style>
    /* Top menu bar background */
    header, .css-1v3fvcr, [data-testid="stHeader"] {
        background-color: #FFFCFE;  /* Light pastel Purple */
    }

    /* Optional: change top menu text color */
    header .css-1d391kg, [data-testid="stHeader"] .css-1d391kg {
        color: #000000;  /* Black text for contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Main text color */
    .stApp, .css-1d391kg, .css-1offfwp {
        color: #1A1A1A;  /* Dark gray text */
    }

    /* Buttons */
    button, .stButton>button {
        background-color: #ADD8E6 !important;  /* Gentle cyan/blue */
        color: #1A1A1A;  /* Dark gray text */
        border-radius: 8px;
        border: none;
    }

    /* Sliders */
    .stSlider>div>div>div>div {
        background-color: #ADD8E6 !important; /* Slider highlight */
    }

    /* Optional: hover effect for buttons */
    button:hover, .stButton>button:hover {
        background-color: #90CAF9 !important; /* Slightly darker blue on hover */
        color: #1A1A1A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Unsupervised ML Playground", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Unsupervised Machine Learning")

# -------------------------------
# Sidebar - Dataset Selection
# -------------------------------
st.sidebar.header("📁 Dataset Options")

uploaded_file = st.sidebar.file_uploader("Upload Your Own CSV", type=["csv"])

sample_data = st.sidebar.selectbox(
    "Or choose sample dataset",
    ["Select from Here", "Iris", "Wine", "Breast Cancer"]
)

# Load data
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif sample_data == "Iris":
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)

elif sample_data == "Wine":
    data = datasets.load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)

elif sample_data == "Breast Cancer":
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)

# -------------------------------
# Data Cleaning
# -------------------------------
if df is not None:

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Keep only numeric columns
    df = df.select_dtypes(include=np.number)

    st.write("Using numeric columns only:", df.columns.tolist())

    # Drop missing values
    df = df.dropna()

    # Scaling option
    scale = st.checkbox("Apply Feature Scaling", value=True)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
    else:
        X = df.values

    # -------------------------------
    # Method Selection
    # -------------------------------
    st.sidebar.header("⚙️ Method Selection")

    method = st.sidebar.selectbox(
        "Choose Unsupervised Method",
        ["K-Means Clustering", "Hierarchical Clustering", "PCA"]
    )

    # =====================================
    # K-MEANS
    # =====================================
    if method == "K-Means Clustering":

        st.header("📍 K-Means Clustering")

        k = st.slider("Number of Clusters (k)", 2, 10, 3)

        if st.button("Run K-Means"):

            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Scatter plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
            ax.set_title("K-Means Clusters (PCA Projection)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)

            # Silhouette score
            sil_score = silhouette_score(X, labels)
            st.success(f"Silhouette Score: {sil_score:.3f}")

        # ---------------------------
        # Elbow + Silhouette Plot
        # ---------------------------
        st.subheader("📈 Model Evaluation")

        ks = range(2, 11)
        wcss = []
        sil_scores = []

        for k_val in ks:
            km = KMeans(n_clusters=k_val, random_state=42)
            km.fit(X)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_))

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(ks, wcss, marker='o')
        ax[0].set_title("Elbow Method")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("WCSS")

        ax[1].plot(ks, sil_scores, marker='o', color='green')
        ax[1].set_title("Silhouette Scores")
        ax[1].set_xlabel("k")

        st.pyplot(fig)

    # =====================================
    # HIERARCHICAL
    # =====================================
    elif method == "Hierarchical Clustering":

        st.header("🌳 Hierarchical Clustering")

        k = st.slider("Number of Clusters", 2, 10, 3)

        linkage_method = st.selectbox(
            "Linkage Method",
            ["ward", "complete", "average"]
        )

        if st.button("Run Hierarchical Clustering"):

            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
            labels = model.fit_predict(X)

            # PCA plot
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
            ax.set_title("Hierarchical Clusters (PCA Projection)")
            st.pyplot(fig)

            sil_score = silhouette_score(X, labels)
            st.success(f"Silhouette Score: {sil_score:.3f}")

        # ---------------------------
        # Dendrogram
        # ---------------------------
        st.subheader("🌲 Dendrogram")

        Z = linkage(X, method=linkage_method)

        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, truncate_mode='level', p=5)
        ax.set_title("Hierarchical Dendrogram")
        st.pyplot(fig)

    # =====================================
    # PCA
    # =====================================
    elif method == "PCA":

        st.header("📉 Principal Component Analysis")

        n_components = st.slider("Number of Components", 2, min(10, X.shape[1]), 2)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        st.write("Explained Variance Ratio:")
        st.write(pca.explained_variance_ratio_)

        # Scatter (first 2 PCs)
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k')
        ax.set_title("PCA Projection")
        st.pyplot(fig)

        # Scree Plot
        st.subheader("📊 Scree Plot")

        pca_full = PCA().fit(X)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)

        fig2, ax2 = plt.subplots()
        ax2.plot(cum_var, marker='o')
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Variance Explained")

        st.pyplot(fig2)

else:
    st.info("Select or Upload your Data, Choose your Machine Learning Method, and then Tune your Parameters.")
    st.image("https://www.mygreatlearning.com/blog/wp-content/uploads/2021/04/ML.jpg")