import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
#import os

#os.chdir("C:\\Users\\Adjie Buanafijar\\Documents\\Adjie\\StarCore\\Klasifikasi Pharmacy Store\\Streamlit")

st.write("""
# Pharmacy Store Prediction App
Aplikasi prediksi **Store Value** dari Apotek
""")

st.sidebar.header('Input Dataset')

st.sidebar.markdown("""
[Contoh file input](https://drive.google.com/file/d/1ZN3_UQP8HVzlH05lhIBUn1mSoRVpxpLn/view)
""")


# Collects user input features into dataframe

uploaded_file = st.sidebar.file_uploader("Upload File", type=["xlsx"])

st.sidebar.header('Input Data Individu')

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

# Reads in saved classification model
load_clf_group = pickle.load(open('DT Store Value Grup (16).pkl', 'rb'))
load_clf_otc = pickle.load(open('DT Store Value OTC (16).pkl', 'rb'))

# Apply model to make predictions
prediction_group = load_clf_group.predict(X)
prediction_otc = load_clf_otc.predict(X)

output_group = pd.DataFrame([input_df["SbjNum"], prediction_group]).T
output_otc = pd.DataFrame([input_df["SbjNum"], prediction_otc]).T

st.subheader('Prediksi Grup')
st.write(output_group)

st.subheader('Prediksi OTC Value')
st.write(output_otc)