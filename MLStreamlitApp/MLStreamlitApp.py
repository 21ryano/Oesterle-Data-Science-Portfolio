## Portfolio Update 3: Machine Learning Application Project

# This is an app, built using Streamlit, to conduct Machine Learning on any dataset!
# First, in "Command Prompt" on the Terminal, type "cd MLStreamlitApp"
# Second, still in "Command prompt", type "streamlit run MLStreamlitApp.py"  

# Import Streamlit/sklearn to build the interactive web application
import streamlit as st
from sklearn import datasets

# Import pandas and numpy for data handling and numerical operations
import pandas as pd
import numpy as np

# Import seaborn and matplotlib for creating visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Import tools for splitting data and evaluating models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Import machine learning models used in this app
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

from pandas.api.types import is_numeric_dtype


# Identify whether data is discrete or continuous
def is_classification_target(y, threshold=50):
    """
    Determines if target should be treated as classification.
    Numeric targets with up to `threshold` unique values are treated as categorical.
    Non-numeric targets are automatically treated as classification.
    """
    if not is_numeric_dtype(y):
        return True
    return y.nunique() <= threshold

# Configure the Streamlit page layout
st.set_page_config(page_title="Machine Learning Playground", layout="wide")

# App Title
st.title("Machine Learning Model App")

# App Description
st.write("""Welcome to the best App for creating Machine Learning Models!""")

st.markdown("""
### How to Use This App:
1. Upload a dataset or choose a sample dataset from the sidebar
2. Select your target variable (what you want to predict)
3. Choose your features (input variables)
4. Select a model and tune hyperparameters
5. Click **Train Model** to see results
""")

# Set the background color to light Purple
st.markdown(
    """
    <style>
    /* Change background color of the entire app */
    .stApp {
        background-color: #FFFCFE;  /* Light pastel Purple */
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
        background-color: #F7F6FF;  /* Pastel purple */
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
        background-color: #A088E0 !important;  /* Gentle cyan/blue */
        color: #1A1A1A;  /* Dark gray text */
        border-radius: 8px;
        border: none;
    }

    /* Sliders */
    .stSlider>div>div>div>div {
        background-color: #A088E0 !important; /* Slider highlight */
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


df = None
target = None
features = None 


# Dataset Selection: In this section, the user can either upload their own dataset or choose a built-in sample dataset. This provides flexibility and allows users to experiment with different data.

# Dataset Selection (Sidebar)
st.sidebar.title("Dataset Options")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV File (Numerical Only)", type=["csv"])

# Sample dataset option
sample_data = st.sidebar.selectbox("Or choose a sample dataset", ["Select from Here", "Titanic", "Iris", "Wine", "Breast Cancer", "Tips", "Penguins", "Diabetes"])

# --- Dataset Selection: Upload CSV or Sample ---
df = None

# Upload CSV
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        uploaded_file.seek(0)  # reset file pointer
        df = pd.read_csv(uploaded_file, encoding='cp1252')
    st.success(f"Dataset uploaded successfully! Shape: {df.shape}")

# Sample datasets
elif sample_data == "Titanic":
    df = sns.load_dataset("titanic")
elif sample_data == "Iris":
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
elif sample_data == "Wine":
    wine = datasets.load_wine(as_frame=True)
    df = wine.frame
elif sample_data == "Breast Cancer":
    bc = datasets.load_breast_cancer(as_frame=True)
    df = bc.frame
elif sample_data == "Diabetes":
    diabetes = datasets.load_diabetes(as_frame=True)
    df = diabetes.frame
elif sample_data == "Tips":
    df = sns.load_dataset("tips")
elif sample_data == "Penguins":
    df = sns.load_dataset("penguins")

# Data Preview/Cleaning
if df is not None:

    # Check for missing values (and Drop Rows with Missing Values)
    st.subheader("Missing Values")
    missing_counts = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_counts.values,
        "Percent Missing": (missing_counts.values / df.shape[0] * 100).round(2)
    })
    
    if missing_counts.sum() > 0:
        st.dataframe(missing_df)
    else:
        st.write("No missing values in the dataset!")

    st.write("Dropping rows with missing values for simplicity.")
    df = df.dropna()
    st.write("Dataset shape after dropping missing values:", df.shape)

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.replace("-", "_").str.replace(" ", "_")



# Feature and Target Selection

# Feature Selection
st.subheader("")

if df is not None:

    # Display columns for feature selection
    columns = df.columns.tolist()

    # Select target variable
    target = st.selectbox("Select Target Variable", columns)

    # Select feature variables
    features = st.multiselect("Select Feature Variables", columns, default=columns[:5])
    if not features:
        st.warning("Please select at least one feature to proceed.")

    # Prevent user from selecting target as a feature
    if target in features:
        features.remove(target)
        st.warning(f"Target variable '{target}' was removed from features.")

# Train-Test Split: Split the dataset into training and testing sets to evaluate performance
if target and features and target not in features:

    X = df[features]
    y = df[target]
    is_classification = is_classification_target(y)
    if is_classification:
        st.success("Detected Classification Problem")
    else:
        st.warning("Detected Regression Problem (continuous target)")

    # Display target variable distribution
    st.subheader("Target Variable Distribution")
    plt.figure(figsize=(8,5))
    if is_classification:
        sns.countplot(x=y)
    else:
        sns.histplot(y, kde=True)
    plt.title(f"Distribution of Target: {target}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.subheader("Train-Test Split")

    # Fixed test size
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write("Training Samples:", X_train.shape[0])
    st.write("Testing Samples:", X_test.shape[0])




# Model Selection and Hyperparameter Tuning
# --- Model Selection Based on User Task Type ---
st.subheader("Choose a Model")

# Let the user choose the task type manually (Classification or Regression)
task_type = st.radio(
    "Choose the type of ML task you want to perform:",
    ["Classification", "Regression"],
    index=0 if is_classification_target(df[target]) else 1
)

is_classification = task_type == "Classification"

# Model selection and hyperparameters
if is_classification:
    # Classification models
    model_name = st.selectbox(
        "Select Classification Model:",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]
    )

    st.subheader("Hyperparameter Tuning")
    if model_name == "Logistic Regression":
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)
    elif model_name == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 10, 4)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_name == "K-Nearest Neighbors":
        k = st.slider("Number of Neighbors (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

else:
    # Regression models
    model_name = st.selectbox(
        "Select Regression Model:",
        ["Linear Regression", "Logistic Regression (for binary targets)"]
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression (for binary targets)":
        # Only allow if target is binary
        if df[target].nunique() == 2:
            C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            model = LogisticRegression(C=C, max_iter=1000)
        else:
            st.warning("Logistic Regression requires a binary target. Falling back to Linear Regression.")
            model = LinearRegression()



# Model Training and Evaluation (automatic)
if df is not None and target and features and target not in features:

    # Train-test split
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=test_size, random_state=42
    )

    from sklearn.preprocessing import StandardScaler

    scale = st.checkbox("Apply Feature Scaling")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if st.button("Train Model"):

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Feature to select 2 Features for KNN Visualization
        if is_classification and model_name == "K-Nearest Neighbors":

            if len(features) >= 2:
                feat1 = st.selectbox("Feature 1 for Visualization", features)
                feat2 = st.selectbox("Feature 2 for Visualization", features, index=1)

                if feat1 == feat2:
                    st.warning("Please select two different features.")
            else:
                st.warning("Need at least 2 features.")


        # KNN Decision Boundary Visualization
        if is_classification and model_name == "K-Nearest Neighbors":
            st.subheader("KNN Decision Boundary")

            if X_train.shape[1] >= 2:
                # Use first 2 features for visualization
                X_vis = X_train[:, :2] if scale else X_train.iloc[:, :2].values
                y_vis = y_train.values

                # Train visualization model
                vis_model = KNeighborsClassifier(n_neighbors=k)
                vis_model.fit(X_vis, y_vis)

                # Create mesh grid
                x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
                y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 200),
                    np.linspace(y_min, y_max, 200)
                )

                grid = np.c_[xx.ravel(), yy.ravel()]
                Z = vis_model.predict(grid)
                Z = Z.reshape(xx.shape)

                # Plot decision boundary
                fig, ax = plt.subplots()
                ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

                # Plot training points
                scatter = ax.scatter(
                    X_vis[:, 0], X_vis[:, 1],
                    c=y_vis,
                    cmap="coolwarm",
                    edgecolors="k"
                )

                ax.set_title(f"KNN Decision Boundary (k={k})")
                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)

                st.pyplot(fig)
            else:
                st.warning("Need at least 2 features to plot KNN decision boundary.")

        if is_classification and model_name == "Logistic Regression":
            st.subheader("Logistic Regression Sigmoid Curve")
        
            # Train model on full feature set
            model = LogisticRegression(C=C, max_iter=1000)
            model.fit(X_train, y_train)
        
            # Let user pick feature to visualize
            feature_name = st.selectbox("Select Feature for Sigmoid Curve", features)
        
            # Get index of selected feature
            feature_index = features.index(feature_name)
        
            # Create a baseline (mean of all features)
            if scale:
                baseline = np.mean(X_train, axis=0)
            else:
                baseline = X_train.mean().values
        
            # Create range for selected feature
            if scale:
                feature_values = X_train[:, feature_index]
            else:
                feature_values = X_train[feature_name].values
        
            x_range = np.linspace(feature_values.min(), feature_values.max(), 200)
        
            # Create input matrix where all features are fixed except one
            X_plot = np.tile(baseline, (len(x_range), 1))
            X_plot[:, feature_index] = x_range
        
            # Predict probabilities
            y_probs = model.predict_proba(X_plot)[:, 1]

            # Get the positive class label
            positive_class = model.classes_[1]
            
            fig, ax = plt.subplots()
            ax.plot(x_range, y_probs, color="red", label=f"P({positive_class})")
            
            # Scatter actual data
            ax.scatter(feature_values, y_train, alpha=0.3, label="Actual Data")
            
            # Updated labels
            ax.set_xlabel(feature_name)  # Feature on X-axis
            ax.set_ylabel(f"Predicted Probability of {target}")  # Show target name
            ax.set_title(f"Effect of {feature_name} on Probability of {positive_class} ({target})")
            
            ax.legend()
            st.pyplot(fig)

            # Metrics with main model
            st.subheader("Model Performance")
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            if len(np.unique(y_test)) == 2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.write("ROC curve is only for binary classification.")


        # Show Decision Tree Visualization
        if is_classification and model_name == "Decision Tree":
            st.subheader("Decision Tree Visualization")

            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=features,
                class_names=[str(c) for c in np.unique(y)],
                filled=True,
                rounded=True,
                fontsize=10
            )
            st.pyplot(fig)

        st.subheader("Model Performance")

        if is_classification:
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.info("Accuracy = percentage of correct predictions.")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # ROC Curve (binary only)
            if len(np.unique(y_test)) == 2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.write("ROC curve is only for binary classification.")

        else:
            # Regression Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R² Score: {r2:.2f}")
            st.info("Reminder: The closer the R² is to 1, the better the model explains the data well.")

            # Actual vs Predicted Plot
            fig3, ax3 = plt.subplots()
            ax3.scatter(y_test, y_pred, alpha=0.7)
            ax3.set_xlabel(f"Actual {target}")
            ax3.set_ylabel(f"Predicted {target}")
            ax3.set_title(f"Actual vs Predicted Values of {target}")
            st.pyplot(fig3)

            # Residual Plot
            residuals = y_test - y_pred
            fig4, ax4 = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax4)
            ax4.set_title(f"Residuals for {target}")
            st.pyplot(fig4)

else:
    st.info("Please upload or select a dataset to begin.")


st.sidebar.markdown("""
**Understanding Data Types:**  
- **Continuous Data**: Numeric values that can take any value in a range (e.g., age, salary, temperature).  
- **Discrete Data**: Numeric or categorical values with distinct options (e.g., number of cars, categories like red/blue/green).  
""")
