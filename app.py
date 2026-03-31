import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# Page configuration
st.set_page_config(page_title="IQROGUEREX - SVM Dashboard", layout="wide")

st.title("⚔️ Support Vector Machine (SVM) Classifier")
st.markdown("### Interactive Dashboard for Linear & Non-Linear Classification")

# --- Load Data ---
@st.cache_data
def load_data():
    file_path = 'Social_Network_Ads.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error("Dataset not found. Please ensure 'Social_Network_Ads.csv' is in the root directory.")
        return None

df = load_data()

if df is not None:
    # --- Sidebar - Model Parameters ---
    st.sidebar.header("SVM Hyperparameters")
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=0)
    c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 25) / 100

    # --- Data Preparation ---
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # --- Model Training ---
    classifier = SVC(kernel=kernel, C=c_val, random_state=0)
    classifier.fit(X_train_scaled, y_train)

    # --- Layout Columns ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Predict User Purchase")
        input_age = st.number_input("Age", min_value=18, max_value=60, value=30)
        input_salary = st.number_input("Estimated Salary", min_value=15000, max_value=150000, value=87000)
        
        if st.button("Classify"):
            prediction = classifier.predict(sc.transform([[input_age, input_salary]]))
            result = "Purchased" if prediction[0] == 1 else "Not Purchased"
            st.info(f"The model predicts: **{result}**")

        # Metrics
        y_pred = classifier.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Not Purchased', 'Purchased'],
                           y=['Not Purchased', 'Purchased'],
                           color_continuous_scale='Blues',
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader(f"SVM Decision Boundary ({kernel.capitalize()} Kernel)")
        
        # Create meshgrid for Plotly contour
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
        y_min, y_max = X[:, 1].min() - 2000, X[:, 1].max() + 2000
        
        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_preds = classifier.predict(sc.transform(grid_points)).reshape(xx.shape)

        fig = go.Figure()

        # Decision Boundary
        fig.add_trace(go.Contour(
            x=x_range, y=y_range, z=grid_preds,
            showscale=False,
            colorscale=[[0, '#FA8072'], [1, '#1E90FF']],
            opacity=0.4, hoverinfo='skip'
        ))

        # Test Set Scatter
        for i, label in enumerate(['Not Purchased', 'Purchased']):
            mask = y_test == i
            fig.add_trace(go.Scatter(
                x=X_test[mask, 0], y=X_test[mask, 1],
                mode='markers', name=label,
                marker=dict(color='#FA8072' if i==0 else '#1E90FF', size=10, line=dict(width=1, color='Black'))
            ))

        fig.update_layout(xaxis_title="Age", yaxis_title="Estimated Salary", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
