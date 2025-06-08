
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('titanic_best_model.pkl')

st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival')

# Input features
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.number_input('Number of Siblings/Spouses', 0, 10, 0)
parch = st.number_input('Number of Parents/Children', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 500.0, 50.0)
embarked = st.selectbox('Embarked', ['Southampton', 'Cherbourg', 'Queenstown'])
family_size = sibsp + parch + 1

# Convert categorical to numerical
sex_num = 0 if sex == 'Male' else 1
embarked_num = 0 if embarked == 'Southampton' else (1 if embarked == 'Cherbourg' else 2)

# Make prediction
if st.button('Predict Survival'):
    # Create input array
    input_data = np.array([[pclass, sex_num, age, sibsp, parch, fare, embarked_num, family_size]])

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f'Passenger would SURVIVE! (Probability: {probability[0][1]:.2%})')
    else:
        st.error(f'Passenger would NOT survive. (Probability: {probability[0][0]:.2%})')
