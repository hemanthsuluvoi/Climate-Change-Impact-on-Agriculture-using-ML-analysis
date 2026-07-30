# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 

import streamlit as st
import base64

 # ------------ TITLE 

st.markdown(f'<h1 style="color:#8d1b92;text-align: center;font-size:36px;">{"Climate Change Impact on Agricultural Land"}</h1>', unsafe_allow_html=True)


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')


data_frame=pd.read_csv("Crop_recommendation.csv")


cloud = sorted(data_frame['label'].unique())

season = data_frame['Season'].unique()

weather_type = data_frame['Weather Type'].unique()


import pickle

with open('model.pickle', 'rb') as f:
    rf = pickle.load(f)
    

# pred_rf = pred_rf

# ================== PREDICTION  ====================

st.write("-----------------------------------------------------------------")
st.write(" Kindly Enter the following details for network log analysis     ")
st.write("------------------------------------------------------=----------")
print()


q1 = st.number_input("Enter the Nitrogen (N)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,format="%.1f")
q2 = st.number_input("Enter the Potassium (K)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,format="%.1f")
q3 = st.number_input("Enter the Phosphorus (P)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,format="%.1f")

q4 = st.number_input("Enter the Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")
q5 = st.number_input("Enter the Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")

q6 = st.number_input("Enter the pH value", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
q7 = st.number_input("Enter the Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")

q8 = st.number_input("Enter the Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=0.0, step=0.1, format="%.1f")
q9 = st.number_input("Enter the Precipitation (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, format="%.1f")
season_1=st.selectbox("Choose Season",season)



def get_alphabetical_position(season):
    # Extract the alphabetical part from the SKU
    alphabet_part = ''.join([char for char in season if char.isalpha()])
    # Calculate the alphabetical position (A=1, B=2, ..., Z=26)
    return sum((ord(char) - ord('A') + 1) for char in alphabet_part)

# Validate the choice
if season_1 in season:
    # Get the alphabetical position of the chosen SKU
    position = get_alphabetical_position(season_1)
    q10=position
    print(f"The alphabetical position of '{season_1}' is: {position}")
else:
    print("Invalid SKU choice.")
    q10=1



weather=st.selectbox("Choose Weather Type",weather_type)



def get_alphabetical_position(weather_type):
    # Extract the alphabetical part from the SKU
    alphabet_part = ''.join([char for char in weather_type if char.isalpha()])
    # Calculate the alphabetical position (A=1, B=2, ..., Z=26)
    return sum((ord(char) - ord('A') + 1) for char in alphabet_part)

# Validate the choice
if weather in weather_type:
    # Get the alphabetical position of the chosen SKU
    position = get_alphabetical_position(weather)
    q11=position
    print(f"The alphabetical position of '{weather}' is: {position}")
else:
    print("Invalid SKU choice.")
    q11=1




butt = st.button("Submit")

if butt:
    if butt:
    # Define inputs list inside the button click
        inputs = [q1, q2, q3, q4, q5, q6, q7, q8, q9]

    # Validation check
        if any(val == 0.0 for val in inputs):
            st.error("Please enter values greater than 0 for all input fields.")
        else:
        # Only when all inputs are non-zero, prediction happens
            data = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11]
            data1 = np.array(data).reshape(1, -1)
            pred_rf = rf.predict(data1)
            pred_rf = int(pred_rf)
        
            if 1 <= pred_rf <= len(cloud):
                result = cloud[pred_rf - 1]
                fin_res = "Identified Crop = " + str(result)
                st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{fin_res}</h1>', unsafe_allow_html=True)

        # elif pred_rf == 1:
        #     st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified Weather = RAINY "}</h1>', unsafe_allow_html=True)
        
        # elif pred_rf == 2:
        #     st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified Weather = SNOWY "}</h1>', unsafe_allow_html=True)
        
        # elif pred_rf == 3:
        #     st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identified Weather = SUUNY "}</h1>', unsafe_allow_html=True)
        