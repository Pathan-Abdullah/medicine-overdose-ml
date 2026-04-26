import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Overdose Risk Prediction", layout="wide")

# ================= HEADER =================
st.markdown("""
# 🩺 Machine Learning Based Clinical Decision Support
### 💊 Medicine Overdose Risk Prediction
""")

MODEL_PATH = "model/production_model.pkl"

# ================= MODEL CHECK =================
if not os.path.exists(MODEL_PATH):
    st.error("🚫 No trained model found. Please train first.")
    st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model_package = load_model()
model = model_package["model"]
encoder = model_package["encoder"]
feature_order = model_package["feature_order"]

# ================= MODEL INFO =================
col1, col2 = st.columns(2)
col1.info(f"📅 Trained On: {model_package.get('trained_on')}")
col2.info(f"📂 Dataset: {model_package.get('dataset_used')}")

st.divider()

# ================= INPUT SECTION =================
st.markdown("## 👤 Patient Details")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 1, 100, 35)

with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])

with c3:
    weight = st.slider("Weight (kg)", 30, 150, 65)

# ================= CONDITIONS =================
st.markdown("## 🧬 Medical Conditions")

c1, c2, c3 = st.columns(3)

with c1:
    diabetes = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")

with c2:
    heart_disease = st.checkbox("Heart Disease")
    kidney_disease = st.checkbox("Kidney Disease")

with c3:
    asthma = st.checkbox("Asthma")
    copd = st.checkbox("COPD")

st.divider()

# ================= MEDICATIONS =================
st.markdown("## 💊 Medications")

if "meds" not in st.session_state:
    st.session_state.meds = []

# Add button
if st.button("➕ Add Medication"):
    st.session_state.meds.append({"drug": "Analgesic", "dosage": 500, "freq": 1})

drug_options = ["Analgesic", "Antibiotic", "Opioid", "Antipyretic", "Sedative"]

# Medication cards
remove_index = None

for i in range(len(st.session_state.meds)):
    st.markdown(f"### Medicine {i+1}")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        drug = st.selectbox("Drug Type", drug_options, key=f"drug{i}")

    with col2:
        dosage = st.number_input(
            "Dosage (mg)",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
            key=f"dose{i}",
        )

    with col3:
        freq = st.selectbox("Frequency / Day", [1, 2, 3, 4], key=f"freq{i}")

    with col4:
        st.write("")
        st.write("")
        if st.button("🗑 Remove", key=f"remove{i}"):
            remove_index = i

    st.session_state.meds[i] = {"drug": drug, "dosage": dosage, "freq": freq}

# Remove medication after loop
if remove_index is not None:
    st.session_state.meds.pop(remove_index)
    st.rerun()

st.divider()

# ================= PREDICTION =================
if st.button("🔍 Predict Risk", use_container_width=True):

    if len(st.session_state.meds) == 0:
        st.error("⚠ Please add at least one medication.")
        st.stop()

    gender_encoded = encoder.transform([gender])[0]

    total_drugs = len(st.session_state.meds)
    total_daily = sum(m["dosage"] * m["freq"] for m in st.session_state.meds)
    max_single = max(m["dosage"] for m in st.session_state.meds)

    # Drug Flags
    flags = {f"{d}_Flag": 0 for d in drug_options}

    for m in st.session_state.meds:
        flags[f"{m['drug']}_Flag"] = 1

    # Input Data
    input_dict = {
        "Age": age,
        "Gender": gender_encoded,
        "Weight": weight,
        "Diabetes": int(diabetes),
        "Hypertension": int(hypertension),
        "Heart_Disease": int(heart_disease),
        "Chronic_Kidney_Disease": int(kidney_disease),
        "Asthma": int(asthma),
        "COPD": int(copd),
        "Total_Drugs": total_drugs,
        "Total_Daily_Dosage": total_daily,
        "Max_Single_Dosage": max_single,
        **flags,
    }

    input_df = pd.DataFrame([input_dict])

    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]

    # Base Prediction
    base_prob = model.predict_proba(input_df)[0][1]

    # ================= INTERACTION RULES =================
    interaction = 0
    reasons = []

    if flags["Opioid_Flag"] and flags["Sedative_Flag"]:
        interaction += 0.25
        reasons.append("Opioid + Sedative combination")

    if total_daily > 4000:
        interaction += 0.20
        reasons.append("High total daily dosage")

    if age > 60 and total_drugs >= 3:
        interaction += 0.15
        reasons.append("Elderly patient with multiple drugs")

    if kidney_disease and total_daily > 3000:
        interaction += 0.20
        reasons.append("Kidney disease with high dosage")

    final_risk = min(base_prob + interaction, 1.0)

    # ================= RESULTS =================
    st.markdown("## 📊 Risk Result")

    r1, r2, r3 = st.columns(3)

    r1.metric("Base Risk", f"{base_prob:.2f}")
    r2.metric("Interaction Score", f"{interaction:.2f}")
    r3.metric("Final Risk", f"{final_risk:.2f}")

    st.progress(final_risk)

    if final_risk > 0.8:
        st.error("🚨 CRITICAL RISK")
    elif final_risk > 0.6:
        st.error("🚨 HIGH RISK")
    elif final_risk > 0.4:
        st.warning("⚠ MODERATE RISK")
    else:
        st.success("✅ LOW RISK")

    # ================= MEDICATION SUMMARY =================
    st.markdown("### 💊 Medication Summary")

    summary_df = pd.DataFrame(st.session_state.meds)
    summary_df.columns = ["Drug Type", "Dosage (mg)", "Frequency/Day"]
    st.dataframe(summary_df, use_container_width=True)

    st.info(f"""
    Total Drugs: {total_drugs}  
    Total Daily Dosage: {total_daily} mg  
    Maximum Single Dose: {max_single} mg
    """)

    # ================= REASONS =================
    if reasons:
        st.markdown("### 🧾 Risk Factors")
        for r in reasons:
            st.write(f"• {r}")

    # ================= DYNAMIC RISK DRIVER CHART =================
    st.markdown("### 📈 Current Patient Risk Factors Overview")

    drivers = {}

    # Age influence
    if age > 60:
        drivers["Age > 60"] = 0.15

    # Multiple drugs
    if total_drugs >= 3:
        drivers["Multiple Drugs"] = 0.12

    # High dosage
    if total_daily > 4000:
        drivers["High Daily Dosage"] = 0.20
    elif total_daily > 2500:
        drivers["Moderate Dosage"] = 0.10

    # Kidney disease
    if kidney_disease:
        drivers["Kidney Disease"] = 0.18

    # Heart disease
    if heart_disease:
        drivers["Heart Disease"] = 0.10

    # Diabetes
    if diabetes:
        drivers["Diabetes"] = 0.06

    # Drug combinations
    if flags["Opioid_Flag"]:
        drivers["Opioid Usage"] = 0.14

    if flags["Sedative_Flag"]:
        drivers["Sedative Usage"] = 0.12

    if flags["Opioid_Flag"] and flags["Sedative_Flag"]:
        drivers["Opioid + Sedative"] = 0.25

    # Low-risk fallback
    if len(drivers) == 0:
        drivers["Low Risk Inputs"] = 0.05

    driver_df = pd.DataFrame(
        list(drivers.items()), columns=["Factor", "Impact"]
    ).sort_values("Impact", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(driver_df["Factor"], driver_df["Impact"])
    ax.set_xlabel("Relative Risk Influence")
    ax.set_title("Patient-Specific Risk Factors")
    plt.tight_layout()
    st.pyplot(fig)
    # ================= SESSION =================
    st.session_state.final_risk = final_risk
    st.session_state.reasons = reasons
