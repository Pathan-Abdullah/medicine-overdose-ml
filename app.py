import streamlit as st

st.set_page_config(
    page_title="Overdose ML System",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
# 💊 Medicine Overdose Detection System

### 🧠 AI-Powered Clinical Decision Support

---

## 🚀 What This System Does

✔ Predicts overdose risk using Machine Learning  
✔ Detects dangerous drug combinations  
✔ Provides risk explanations  
✔ Supports safer medical decisions  

---

## 🧭 Navigation

Use the **sidebar** to:
- Train Model 📊
- Predict Risk 🔍
- View Precautions ⚠️

---

### ⚠ Disclaimer
This is a **decision-support system**, not a replacement for doctors.
""")