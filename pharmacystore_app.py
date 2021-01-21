import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
#import os

#os.chdir("C:\\Users\\Adjie Buanafijar\\Documents\\Adjie\\StarCore\\Klasifikasi Pharmacy Store\\Streamlit")

st.write("""
# Pharmacy Store Prediction App
This app predicts the **Store Value Group** !
""")

st.sidebar.header('Input Data')

st.sidebar.markdown("""
[Contoh file input](https://drive.google.com/file/d/1ZN3_UQP8HVzlH05lhIBUn1mSoRVpxpLn/view)
""")


# Collects user input features into dataframe

uploaded_file = st.sidebar.file_uploader("Upload File", type=["xlsx"])
if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)
    X = input_df.drop("SbjNum", axis=1)
else:
    def user_input_features():
        sbjnum = st.sidebar.text_input('SbjNum', value="000000000")
        jml_pekerja = st.sidebar.text_input('Jumlah Pekerja',value="10")
        jml_sku_otc = st.sidebar.text_input('Jumlah SKU OTC',value="10")
        luas_shelf_total = st.sidebar.text_input('Luas Shelf Total',value="10")
        data = {"SbjNum" : sbjnum,
                "Jml_pekerja" : jml_pekerja,
                "jumlah SKU OTC" : jml_sku_otc,
                "Luas Shelf Total" : luas_shelf_total}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    X = input_df.drop("SbjNum", axis=1)

# PREDICT GROUP

# Reads in saved classification model
load_clf_group = pickle.load(open('DT Store Value Grup (16).pkl', 'rb'))

# Apply model to make predictions
prediction_group = load_clf_group.predict(X)
prob_group = load_clf_group.predict_proba(X)

st.subheader('Prediksi Grup')
st.write(prediction_group)

st.subheader('Peluang Grup')
st.write(prob_group)

# PREDICT OTC VALUE
# Reads in saved classification model
load_clf_otc = pickle.load(open('DT Store Value OTC (16).pkl', 'rb'))

# Apply model to make predictions
prediction_otc = load_clf_otc.predict(X)

st.subheader('Prediksi OTC Value')
st.write(prediction_otc)

#st.subheader('Prediction Probability')
#st.write(prob_group)
