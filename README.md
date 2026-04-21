# Medicine Overdose Risk Predictor

A deployed machine learning web application that predicts medicine overdose risk using patient information, medication details, and drug interaction logic.

## Live Demo
https://medicine-overdose-predictor.streamlit.app/

## Features
- Predict overdose risk probability
- Supports multiple medications
- Hybrid ML + rule-based interaction logic
- Risk classification (Low / Moderate / High / Critical)
- Personalized precautions
- Explainable outputs

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

## Machine Learning Model
- Random Forest Classifier
- Balanced class weighting
- Feature engineered inputs

## How to Run Locally

pip install -r requirements.txt
streamlit run app.py

## Disclaimer
This project is for educational and decision-support purposes only. Not medical advice.
