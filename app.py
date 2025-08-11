# student_recommendation_app.py
import streamlit as st
import pandas as pd
import pickle

# --------------------
# Load saved model, scaler, and label encoder
# --------------------
with open("nb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label.pkl", "rb") as f:
    label_enc = pickle.load(f)

# --------------------
# App title
# --------------------
st.title("🎓 Student Recommendation Prediction System")
st.write("Enter the student’s details to predict whether they will receive a recommendation.")

# --------------------
# User inputs
# --------------------
overall_grade = st.selectbox(
    "Overall Grade",
    options=["A", "B", "C", "D", "E", "F"]
)

obedient = st.selectbox(
    "Obedient (Follows Rules & Instructions)",
    options=["Yes", "No"]
)

research_score = st.number_input(
    "Research Score (0-100)",
    min_value=0, max_value=100, value=85
)

project_score = st.number_input(
    "Project Score (0-100)",
    min_value=0, max_value=100, value=88
)

# --------------------
# Map grades to numeric
# --------------------
grade_mapping = {'A': 90, 'B': 80, 'C': 70, 'D': 60, 'E': 50, 'F': 40}
overall_grade_num = grade_mapping[overall_grade]

# Encode obedient field
obedient_num = label_enc.transform([obedient])[0]

# --------------------
# Prediction button
# --------------------
if st.button("Predict Recommendation"):
    # Create dataframe for new student
    new_student = pd.DataFrame({
        'OverallGrade': [overall_grade_num],
        'Obedient': [obedient_num],
        'ResearchScore': [research_score],
        'ProjectScore': [project_score]
    })

    # Scale features
    new_student_scaled = scaler.transform(new_student)

    # Predict
    recommendation = model.predict(new_student_scaled)[0]

    # Show result
    if recommendation == 1:
        st.success("✅ This student is **Recommended**.")
    else:
        st.error("❌ This student is **Not Recommended**.")

    st.write(f"Prediction Value: **{recommendation}** (1=Yes, 0=No)")
