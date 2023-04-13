import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.model_selection import validation_curve, LeaveOneOut, train_test_split, cross_val_score
from sklearn.model_selection import cross_validate, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder 
from matplotlib import pyplot
import pickle
import math

#Baca Data
data = pd.read_excel("./data_modeling1.xlsx")
raw = pd.read_excel("./data_modeling.xlsx")
loaded_model = pickle.load(open('./model.pkl', 'rb'))
dict = {"ACEH":0, "SUMATERA UTARA":33, "SUMATERA BARAT":31, "RIAU":25, "JAMBI":7,
"SUMATERA SELATAN":32, "BENGKULU":3, "LAMPUNG":18, "KEPULAUAN BANGKA BELITUNG":16,
"KEPULAUAN RIAU": 17, "DKI JAKARTA": 5, "JAWA BARAT":8, "JAWA TENGAH":9, 
"DI YOGYAKARTA":4, "JAWA TIMUR": 10, "BANTEN":2, "BALI":1, "NUSA TENGGARA BARAT":21,
"NUSA TENGGARA TIMUR":22, "KALIMANTAN BARAT":11, "KALIMANTAN TENGAH": 13,
"KALIMANTAN SELATAN": 12, "KALIMANTAN TIMUR":14, "KALIMANTAN UTARA":15, "SULAWESI UTARA":30,
"SULAWESI TENGAH":28, "SULAWESI SELATAN":27, "SULAWESI TENGGARA":29, "GORONTALO":6,
"SULAWESI BARAT":26, "MALUKU":19, "MALUKU UTARA":20, "PAPUA BARAT":24,"PAPUA":23}

LOGO_IMAGE = "./logo.jpeg"
#Disable Warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#Set Size
sns.set(rc={'figure.figsize':(8,8)})
#Coloring
colors_1 = ['#66b3ff','#99ff99']
colors_2 = ['#66b3ff','#99ff99']
colors_3 = ['#79ff4d','#4d94ff']
colors_4 = ['#ff0000','#ff1aff']
st.markdown(
    f"""
    <div style="text-align: center;">
    <img class="logo-img" src="data:png;base64,{base64.b64encode(open(LOGO_IMAGE, 'rb').read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #243A74; font-family:sans-serif'>ANALISIS PREVALENSI STUNTING</h1>", unsafe_allow_html=True)
menu = st.sidebar.selectbox("Select Menu", ("Dashboard", "Prediksi"))
if menu == "Dashboard":
    st.write("Menu Dashboard")
    st.write("## Link ")
if menu == "Prediksi":
    st.write("Menu Prediksi")
    df = pd.DataFrame(dict,index=[0])
    df = df.transpose()
    st.write(data.head(2))
    provinsi = st.selectbox("Pilih Provinsi",raw['PROVINSI'].unique())
    for item in raw['PROVINSI'].unique():
         if item == provinsi:
             st.write(' Provinsi yang dipilih adalah ', str(provinsi))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Kondisi IPM</p>", unsafe_allow_html=True)
        
        input_ipm = st.number_input('Nilai IPM',key=1, value= raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['IPM'].values[0])
        for item3 in raw['IPM'].unique():
            if item == provinsi and raw['Tahun'] == 2022:
                st.write(input_ipm)
        
        
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>PDRB per Kapita (juta rupiah)  </p>", unsafe_allow_html=True)
        input_pdrb = st.number_input('PDRB per Kapita (dalam juta rupiah)',key=2,value= raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['PDRB_perkapita_juta'].values[0])
        for item4 in raw['PDRB_perkapita_juta'].unique():
            if item == provinsi and raw['Tahun'] == 2022:
                st.write(input_pdrb)
    with col2:
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>Persentase Akses ke Faskes</p>", unsafe_allow_html=True)
        
        input_unmeet = st.number_input('UNMEET NEEDS',key=3, value= raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['UNMET_NEED'].values[0])
        for item3 in raw['UNMET_NEED'].unique():
            if item == provinsi and raw['Tahun'] == 2022:
                st.write(input_unmeet)
        
        
        st.markdown("<p style='text-align: center; color: #FFCC29; font-family:arial'>DAK Gabungan (dalam Miliar Rupiah)</p>", unsafe_allow_html=True)
        input_dak = st.number_input('DAK Gabungan (dalam Miliar Rupiah)',key=4,value= raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['DAK_10M'].values[0])
        for item4 in raw['DAK_10M'].unique():
            if item == provinsi and raw['Tahun'] == 2022:
                st.write(input_dak)
        
    st.write("Tingkat Prevalensi Tahun 2023 pada Provinsi : ",str(provinsi))
    st.write("## ", raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['PREVALENSI'].values[0])
    
    if st.button("Prediksi"):
        st.write("Prediksi Sukses")
        #define X & y
        X = data.drop(['PREVALENSI','Tahun'], axis=1)
        y = data['PREVALENSI']
        index=[0]
        for item in df.index:
            if item == provinsi:
                provinsi_enc = df.loc[provinsi].values[0]
        df_1_pred = pd.DataFrame({
            'PROVINSI' : provinsi_enc,
            'IPM' : input_ipm,
            'PDRB_log' : input_pdrb,
            'UN_sqrt' : input_unmeet,
            'DAK_sqrt' : input_dak,
        },index=index)
        #Set semua nilai jadi 0
        df_kosong_1 = X[:1]
        for col in df_kosong_1.columns:
            df_kosong_1[col].values[:] = 0
        list_1 = []
        for i in df_1_pred.columns:
            x = df_1_pred[i][0]
            list_1.append(x)
        #buat dataset baru
        for i in df_kosong_1.columns:
            for j in list_1:
                if i == j:
                    df_kosong_1[i] = df_kosong_1[i].replace(df_kosong_1[i].values,1)  
        df_kosong_1['PROVINSI'] = df_1_pred['PROVINSI']   
        df_kosong_1['IPM'] = df_1_pred['IPM']
        df_kosong_1['PDRB_log'] = math.log(df_1_pred['PDRB_log'])
        df_kosong_1['UN_sqrt'] = math.sqrt(df_1_pred['UN_sqrt'])
        df_kosong_1['DAK_sqrt'] = math.sqrt(df_1_pred['DAK_sqrt'])
        pred_1 = loaded_model.predict(df_kosong_1)
        prevalensipred = raw[(raw['PROVINSI'] == provinsi) & (raw['Tahun'] == 2022)]['PREVALENSI'].values[0]
        pred_selisih = pred_1 - prevalensipred
        st.write('Prediksi Prevalensi berdasar data diatas adalah : ')
        st.write('{0:.2f}'.format(pred_1[0]))

        st.write('Pravalensi tahun 2022 berdasar data diatas adalah : ')
        st.write('{0:.2f}'.format(prevalensipred))

        st.write('Selisih Prediksi Prevalensi berdasar data diatas adalah : ')
        st.write('{0:.2f}'.format(pred_selisih[0]))