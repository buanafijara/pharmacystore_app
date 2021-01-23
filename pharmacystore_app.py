import streamlit as st
import pandas as pd
import pickle
import base64
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

output = pd.DataFrame([input_df["SbjNum"], prediction_otc, prediction_group]).T
output.columns = ["SbjNum", "Pred OTC Value", "Pred Grup"]
output.set_index("SbjNum", inplace=True)

st.subheader('Prediksi Grup')
st.write(output)

# Download Button
def filedownload(df):
    csv = df.to_csv(sep = ",")
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="pred_pharmacystore.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(output), unsafe_allow_html=True)
