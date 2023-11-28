#Import libraries needed
from ast import Div
import base64
import json
from turtle import width
import streamlit as st
import joblib
import numpy as np
from PIL import Image


# Sidebar
st.sidebar.write("""# *Breast Cancer Diagnosis by Machine Learning* """)

language = st.sidebar.selectbox('Select language', ('EN', 'ES'))

with open('/Users/danirubio/desktop/languages/%s.json' % language, 'r', encoding="utf-8") as translation_file:    
    translation = json.load(translation_file)

st.sidebar.write(translation['project_description'])

# Sidebar footer
st.sidebar.write("""
	%s **Daniela Rubio** \n
	*© 2023 [Autonomous University of Chihuhua](https://uach.mx)*
	""" % translation['developed_by'])


st.write("""## %s""" % (translation['title']))
st.write("""## %s""" % (translation['actual_situation']))

col1, col2 = st.columns(2)
with col1:
    """%s""" % translation['introduction1']
    

with col2:
    st.image(
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTd_ir0MY9vFFfgXox_KpbEF2khz3ulGi5VEQ&usqp=CAU',
         caption=(translation['fig'])
    )

#Load the pipeline and the Logistic Regression Model
pipeline = joblib.load('/Users/danirubio/desktop/pipeline_transformer.sav')
LR_model = joblib.load('/Users/danirubio/desktop/LR_model.sav')
SVM_model = joblib.load('/Users/danirubio/desktop/SVM_model.sav')
DT_model = joblib.load('/Users/danirubio/desktop/DT_model.sav')
VC_model = joblib.load('/Users/danirubio/desktop/VC_model.sav')

st.write('---')

#Some text as instructions in the app
st.text(""" %s""" % (translation['input_instructions']))


col3, col4 = st.columns(2)
with col3:
    area_mean = float(st.number_input(translation['area_mean'], min_value=0.0, max_value=100000.0))
    compactness_mean = float(st.number_input(translation['compactness_mean'], min_value=0.0, max_value=100000.0))
    concavity_mean = float(st.number_input(translation['concavity_mean'],min_value=0.0, max_value=100000.0))
    

with col4:
    concave_points_mean = float(st.number_input(translation['concave_pm'],min_value=0.0, max_value=100000.0))
    area_se = float(st.number_input('Area se',min_value=0.0, max_value=100000.0))
    model_option = st.selectbox(translation['select_model'], ["Logistic Regression","Support Vector Machine","Decision Tree","Ensemble"])


#Function to get the prediction taking in consideration the model chosen
#and the features' inputs
def get_prediction(model, features):
    features = pipeline.transform(features)
    prediction = model.predict(features)
    return prediction

#Create button that will classify and trigger the prediction
btn = st.button(translation['classify'])

st.title(translation['tumor_is'])


#Gets features' inputs, prediction and the result
if btn:
    features = ([[area_mean, compactness_mean, concavity_mean, concave_points_mean,area_se]])
    if model_option== "Support Vector Machine":
        prediction = get_prediction(SVM_model, features)
    elif model_option == "Logistic Regression":
        prediction = get_prediction(LR_model, features)
    elif model_option == "Decision Tree":
        prediction = get_prediction(DT_model, features)
    elif model_option == "Ensemble":
        prediction = get_prediction(VC_model, features)    
    result = (translation['Malign']) if prediction[0] == 1 else (translation['Benign'])
    st.write(f"{result}")

