import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Model Training", layout="wide")

st.markdown("# 📊 Model Training Dashboard")
st.caption("Train ML model with dataset and evaluate performance")

MODEL_PATH = "model/production_model.pkl"
os.makedirs("model", exist_ok=True)

EXPECTED_COLUMNS = [
    "Age","Gender","Weight",
    "Diabetes","Hypertension","Heart_Disease",
    "Chronic_Kidney_Disease","Asthma","COPD",
    "Total_Drugs","Total_Daily_Dosage","Max_Single_Dosage",
    "Opioid_Flag","Sedative_Flag","Antibiotic_Flag",
    "Analgesic_Flag","Antipyretic_Flag","Overdose"
]

# ================= MODEL INFO =================
st.markdown("## 📦 Model Status")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_package = pickle.load(f)

    c1, c2 = st.columns(2)
    c1.success("Model Available")
    c2.info(f"Dataset: {model_package.get('dataset_used')}")

    st.info(f"📅 Trained On: {model_package.get('trained_on')}")
else:
    st.warning("No model found. Train a new model.")

st.divider()

# ================= DATA UPLOAD =================
st.markdown("## 📂 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if list(df.columns) != EXPECTED_COLUMNS:
        st.error("Dataset schema mismatch!")
        st.stop()

    df = df.drop_duplicates().fillna(0)
    df = df[df["Total_Daily_Dosage"] < 10000]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", len(EXPECTED_COLUMNS)-1)

    if st.button("👀 Preview Dataset"):
        st.dataframe(df.head(20))

    # ================= TRAIN =================
    if st.button("🚀 Train Model", use_container_width=True):

        df_processed = df.copy()

        le = LabelEncoder()
        df_processed["Gender"] = le.fit_transform(df_processed["Gender"])

        X = df_processed.drop("Overdose", axis=1)
        y = df_processed["Overdose"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.markdown("## 📊 Model Performance")

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        model_package = {
            "model": model,
            "encoder": le,
            "feature_order": list(X.columns),
            "trained_on": str(datetime.now()),
            "dataset_used": uploaded_file.name
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_package, f)

        st.success("✅ Model trained and saved successfully.")