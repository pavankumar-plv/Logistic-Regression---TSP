import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained pipeline
with open("model_pipeline.pkl", "rb") as model_file:
    model_pipeline = pickle.load(model_file)

st.title("Titanic Survival Prediction App")

st.write("Enter passenger details to predict survival.")

# Input fields for user
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["Male", "Female"]) 

# Encode categorical variables
sex_encoded = 1 if sex == "Male" else 0

# Prepare input data for prediction
input_data = pd.DataFrame([[pclass, age, fare, sibsp, parch, sex]],
                          columns=["Pclass", "Age", "Fare", "SibSp", "Parch", "Sex"])

# Make prediction
if st.button("Predict Survival"):
    prediction = model_pipeline.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.write(f"Prediction: {result}")

    # Display survival probability
    probability = model_pipeline.predict_proba(input_data)[0][1]
    st.write(f"Survival Probability: {probability:.2f}")

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x=["Did Not Survive", "Survived"], y=model_pipeline.predict_proba(input_data)[0], ax=ax)
    ax.set_ylabel("Probability")
    ax.set_title("Survival Probability Distribution")
    st.pyplot(fig)

# Display dataset insights
st.subheader("Survival Rate by Class")
data = pd.read_csv('Titanic_train.csv')
fig, ax = plt.subplots()
sns.barplot(x='Pclass', y='Survived', data=data, ax=ax)
ax.set_ylabel("Survival Rate")
st.pyplot(fig)

st.subheader("Survival Rate by Sex")
fig, ax = plt.subplots()
sns.barplot(x='Sex', y='Survived', data=data, ax=ax)
ax.set_ylabel("Survival Rate")
st.pyplot(fig)
