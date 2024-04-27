import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# %pip install h5py graphviz pydot


data = pd.read_csv('Lung_Cancer_Dataset.csv')
data = data.rename(columns={'FATIGUE ': 'FATIGUE', 
                        'ALLERGY ': 'ALLERGY'})
data.head()
df = data.copy()
ds = data.copy()

df['GENDER'] = df['GENDER'].replace('M', 1)
df['GENDER'] = df['GENDER'].replace('F', 0)
df['GENDER'] = df['GENDER'].astype(int)
# ds.info()



# #using XGBOOST to find feature importance
# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x,y)

# # first feature importance scores
# xgb.plot_importance(model)


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# x_train, x_test, y_train, y_test = train_test_split(new_ds, y, test_size = 0.10, random_state = 47, stratify = y)
# print(f'x_train: {x_train.shape}')
# print(f'x_test: {x_test.shape}')
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))


# DEEPE LEARNING MODEL

# model = tf.keras.Sequential([ 
#     tf.keras.layers.Dense(units=12, activation='relu'),
#     tf.keras.layers.Dense(20, activation='relu'), 
#     tf.keras.layers.Dense(20, activation='relu'), 
#     tf.keras.layers.Dense(1, activation='sigmoid') 
# ])
# model.compile(optimizer='adam',
#               loss = 'binary_crossentropy', 
#               metrics=['accuracy']) 

# model.fit(x_train, y_train, epochs=15) 

# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import classification_report

# revealer = confusion_matrix(y_pred, y_test)
# sns.set(style = 'darkgrid')
# sns.heatmap(revealer/np.sum(revealer), annot=True, cmap='crest', fmt='.1%', linewidth=1)

# print(classification_report(y_pred, y_test))

# model.save('heartfailurepred.h5')




import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('cavistalungcancerspred.h5')

# st.sidebar.image('pngwing.com.png', width = 200,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Lung Cancer Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>Lung cancer is one of the most common and deadliest forms of cancer worldwide. Early detection and intervention are crucial for improving patient outcomes. This project aims to develop a machine learning model that predicts the likelihood of an individual having lung cancer based on various health-related factors.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('anatomical-images-human-lungs.jpg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)



    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>FATIGUE</h3>", unsafe_allow_html=True)
    st.markdown("<p>FATIGUE refers to the biological FATIGUE of the individual, which can have an impact on their susceptibility to diabetes. There are three</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>The Age field in the Heart Diseases Model denotes the individual's chronological age, a key factor in assessing heart disease risk. Age ranges from 28-77 in our dataset.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>CHRONIC DISEASE </h3>", unsafe_allow_html=True)
    st.markdown("<p>CHRONIC DISEASE is a type of fat that is found in your blood. It's essential for building cells and making certain hormones, but having too much CHRONIC DISEASE can be harmful, especially for your heart. Levels between 100-129 mg/dL are considered near optimal, while Levels above 160 mg/dL are considered high.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>MAXHR </h3>", unsafe_allow_html=True)
    st.markdown("<p>MAXHR stands for Maximum Heart Rate. It refers to the highest heart rate an individual can achieve during physical exertion. MAXHR is an important metric in assessing cardiovascular health and fitness levels.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Chest pain type</h3>", unsafe_allow_html=True)
    st.markdown("<p>Chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>ALLERGY</h3>", unsafe_allow_html=True)
    st.markdown("<p>Resting blood pressure refers to the measurement of blood pressure when the body is at rest, typically in a seated or lying position and after a period of relaxation. Normal resting blood pressure typically falls below 120/80 mmHg</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>GENDER </h3>", unsafe_allow_html=True)
    st.markdown("<p>GENDER is a term used when doctors are checking the heart's health. They might ask you to exercise on a treadmill or bike while they monitor your heart. They're looking for any unusual changes on a graph called an electrocardiogram (ECG). One thing they look for is if a part of the graph dips down (depression) when you're exercising compared to when you're resting. This dip is called GENDER. If there's a big dip, it could mean your heart isn't getting enough blood, which could signal a problem like blocked arteries.</p>", unsafe_allow_html=True)

   

    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by the Orpheus Snipers at the Cavista Hackathon </p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(ds[['AGE', 'FATIGUE', 'CHRONIC DISEASE', 'SHORTNESS OF BREATH', 'YELLOW_FINGERS', 
                    'SMOKING', 'GENDER', 'ALCOHOL CONSUMING', 'CHEST PAIN', 'COUGHING', 
                    'ALLERGY', 'SWALLOWING DIFFICULTY', 'ANXIETY']])



if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    st.sidebar.markdown("Add your input here")
    AGE = st.sidebar.number_input("AGE",0,1000)
    FATIGUE = st.sidebar.selectbox("FATIGUE", df['FATIGUE'].unique())
    # chronic_disease = st.sidebar.number_input("CHRONIC DISEASE", df['CHRONIC DISEASE'].unique())
    # ds['CHRONIC DISEASE'] = chronic_disease
    ANXIETY = st.sidebar.selectbox("ANXIETY", df['ANXIETY'].unique())
    YELLOW_FINGERS = st.sidebar.selectbox("YELLOW_FINGERS", df['YELLOW_FINGERS'].unique())
    ALLERGY = st.sidebar.selectbox("ALLERGY", df['ALLERGY'].unique())
    GENDER = st.sidebar.selectbox("GENDER", df['GENDER'].unique())
    COUGHING = st.sidebar.selectbox("COUGHING", df['COUGHING'].unique())
    SMOKING = st.sidebar.selectbox("SMOKING", df['SMOKING'].unique())
    CHRONIC_DISEASE = st.sidebar.selectbox("CHRONIC DISEASE", df['CHRONIC DISEASE'].unique())
    SHORTNESS_OF_BREATH = st.sidebar.selectbox("SHORTNESS OF BREATH", df['SHORTNESS OF BREATH'].unique())
    ALCOHOL_CONSUMING = st.sidebar.selectbox("ALCOHOL CONSUMING", df['ALCOHOL CONSUMING'].unique())
    CHEST_PAIN = st.sidebar.selectbox("CHEST PAIN", df['CHEST PAIN'].unique())
    SWALLOWING_DIFFICULTY = st.sidebar.selectbox("SWALLOWING DIFFICULTY", df['SWALLOWING DIFFICULTY'].unique())




    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    patient_name = st.text_input("Patient Name")



    input_variables = pd.DataFrame([{
        'AGE': AGE,
        'FATIGUE':FATIGUE,
        'ANXIETY': ANXIETY, 
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ALLERGY': ALLERGY,
        'GENDER': GENDER,
        'COUGHING': COUGHING,
        'SMOKING': SMOKING,
        'CHRONIC DISEASE': CHRONIC_DISEASE,
        'SHORTNESS OF BREATH': SHORTNESS_OF_BREATH,
        'ALCOHOL CONSUMING': ALCOHOL_CONSUMING,
        'CHEST PAIN': CHEST_PAIN,
        'SWALLOWING DIFFICULTY': SWALLOWING_DIFFICULTY
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)


    if st.button('Press To Predict'):
        st.markdown("<h4 style = 'color: #2B2A4C; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)
        predicted = model.predict(input_variables)
        st.toast('Predicted Successfully')
        st.image('check icon.png', width = 100)
        st.success(f'Model Predicted {int(np.round(predicted))}')
        if predicted >= 0.5:
            st.error('High risk of Lung Cancer!')
        else:
            st.success('Low risk of  Lung Cancer.')
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>Lung Cancer Prediction MODEL BUILT BY The Orpheus Snipers at the Cavista Hackathon 2024</h8>",unsafe_allow_html=True)
