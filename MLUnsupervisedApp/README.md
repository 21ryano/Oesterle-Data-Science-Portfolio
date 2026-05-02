# Unsupervised Machine Learning Streamlit App
This is an interactive Streamlit app that allows users to explore **unsupervised machine learning techniques** on any numerical dataset. Users can upload their own data or select a built-in sample dataset, then apply clustering and dimensionality reduction methods to uncover hidden patterns in the data. The goal of this project is to make an easy-to-use unsupervised learning application with easy to interpret visualizations.

 **How To Use**

- Launch the app locally or deploy it using Streamlit Community Cloud.

### If running locally:
- Clone the repository: **git clone https://github.com/21ryano/Oesterle-Data-Science-Portfolio.git**
- Navigate to the app folder: **cd MLStreamlitApp**
- Install required libraries: **pip install Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, Scipy, and Streamlit**
- Run the app: **streamlit run app.py**


### APP FEATURES
- **Upload your own dataset or choose a built-in sample dataset** (Iris, Wine, Breast Cancer)
- Choose between unsupervised learning methods: **K-Means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA)**
- **Hyperparameter tuning:** *Number of clusters (k)*, *Linkage method (ward, complete, average)*, and *Number of PCA components*
- **Model evaluation tools:** *Elbow Method (WCSS)*, *Silhouette Score*, *Dendrogram visualization*, *PCA scatterplots*, and *Explained variance (scree plot)*
- Automatic preprocessing: Handles missing values, Uses only numeric features, and Optional feature scaling


### Key Learning Outcomes
- Understanding how clustering algorithms group similar data points
- Exploring how different values of K affect clustering results
- Interpreting dendrograms for hierarchical clustering
- Using PCA for dimensionality reduction and visualization
- Evaluating unsupervised learning with silhouette scores and variance explained


### References

- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Streamlit Documentation: https://docs.streamlit.io/
- SciPy Hierarchical Clustering: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html


### Visual Examples
![alt text](https://github.com/21ryano/Oesterle-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Screenshots/Screenshot%202026-05-02%20130841.png)
![alt text](https://github.com/21ryano/Oesterle-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Screenshots/8c553f8095300b28bb1e0ca251f245cba5e528656ed3687100768490.png)
![alt text](https://github.com/21ryano/Oesterle-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Screenshots/Screenshot%202026-05-02%20130441.png)
