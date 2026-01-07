import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Wine Clustering App", layout="centered")

st.title("🍷 Wine Clustering App (DBSCAN)")
st.write("Enter wine chemical properties to find the cluster")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("wine_clustering_data.csv")

features = [
    "alcohol", "malic_acid", "ash", "ash_alcanity", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280", "proline"
]

X = df[features]

# -------------------------------
# Scale + Train DBSCAN
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=1.5, min_samples=5)
clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔢 Enter Wine Sample Values")

user_input = []

for col in features:
    value = st.number_input(col.replace("_", " ").title(), value=0.0)
    user_input.append(value)

# -------------------------------
# Predict Cluster
# -------------------------------
if st.button("Find Cluster"):
    user_scaled = scaler.transform([user_input])
    user_cluster = model.fit_predict(
        np.vstack([X_scaled, user_scaled])
    )[-1]

    if user_cluster == -1:
        st.error("❌ This wine sample is considered NOISE (Outlier)")
    else:
        st.success(f"✅ This wine belongs to Cluster: {user_cluster}")

# -------------------------------
# Show Dataset Clusters
# -------------------------------

