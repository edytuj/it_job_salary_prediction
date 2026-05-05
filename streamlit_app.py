import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title(f"IT Salary Predictor")

st.write("Enter job details to estimate salary.")

# Inputs

title = st.text_input("Job title", placeholder="e.g. Python Developer")

skills_input = st.text_input(
    "Skills (comma separated)", placeholder="e.g. python, aws, docker"
)

city = st.text_input("City", placeholder="e.g. Warszawa")

seniority = st.selectbox("Seniority", ["junior", "mid", "senior"])

# Action

if st.button("Predict salary"):
    if not title or not skills_input or not city:
        st.error("Please fill in all fields.")
    else:
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]

        payload = {
            "title": title,
            "skills": skills,
            "city": city,
            "seniority": seniority,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 200:
                data = response.json()

                st.success("Prediction ready!")

                st.subheader("Estimated Salary")
                st.write(data["prediction"])

                st.subheader("Range")
                st.write(f"{data['range'][0]} - {data['range'][1]}")

                st.subheader("Uncertainty")
                st.write(data["uncertainty"])

                st.subheader("Confidence")
                st.write(f"Absolute: {data['confidence_absolute']}")
                st.write(f"Relative: {data['confidence_relative']}")

            elif response.status_code == 422:
                st.error("Invalid input. Please check your data.")

            else:
                st.error(f"Error: {response.text}")

        except Exception as e:
            st.error(f"Could not connect to API: {e}")
