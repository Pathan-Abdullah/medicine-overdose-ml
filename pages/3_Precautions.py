import streamlit as st

st.set_page_config(page_title="Precautions", layout="wide")

st.markdown("# ⚠️ Smart Medication Safety Advisor")

# ================= CHECK SESSION =================
if "final_risk" not in st.session_state:
    st.warning("⚠ Please run a prediction first.")
    st.stop()

risk = st.session_state.get("final_risk", 0)
reasons = st.session_state.get("reasons", [])

# ================= RISK HEADER =================
st.markdown("## 🧠 Risk-Based Guidance")

if risk > 0.8:
    st.error("🚨 CRITICAL RISK - Immediate action required")
elif risk > 0.6:
    st.error("🚨 HIGH RISK - Medical attention advised")
elif risk > 0.4:
    st.warning("⚠ MODERATE RISK - Monitor closely")
else:
    st.success("✅ LOW RISK - Safe with precautions")

st.progress(risk)

# ================= GENERAL PRECAUTIONS =================
st.markdown("## 🛡 General Safety Measures")

general = [
    "Do not exceed prescribed dosage",
    "Follow timing strictly",
    "Avoid self-medication",
    "Consult doctor before combining drugs"
]

for g in general:
    st.write(f"✔ {g}")

# ================= DYNAMIC RULES =================
st.markdown("## 🔬 Personalized Risk Factors")

if reasons:
    for r in reasons:
        if "Opioid" in r:
            st.warning("Avoid combining opioids with sedatives → can cause respiratory depression")

        if "dosage" in r.lower():
            st.warning("Reduce total daily dosage → high toxicity risk")

        if "Elderly" in r:
            st.warning("Elderly patients require lower dosages and close monitoring")

        if "Kidney" in r:
            st.warning("Kidney condition → drug clearance reduced → toxicity risk")

# ================= EXTRA INTELLIGENCE =================
st.markdown("## 💡 Smart Recommendations")

if risk > 0.6:
    st.error("""
    - Immediate medical consultation required  
    - Consider reducing number of medications  
    - Monitor vital signs regularly  
    """)

elif risk > 0.4:
    st.warning("""
    - Adjust dosage if possible  
    - Avoid risky drug combinations  
    - Regular monitoring advised  
    """)

else:
    st.success("""
    - Continue prescribed medications  
    - Maintain healthy routine  
    - Stay hydrated  
    """)

# ================= DISCLAIMER =================
st.markdown("---")
st.caption("⚠ This system supports decisions and does not replace professional medical advice.")