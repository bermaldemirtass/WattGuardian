import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Özel stil ayarları
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h3 {
            text-align: center;
        }

        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.5em 1em;
            font-weight: bold;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #144c7c;
        }

        .stSlider > div[data-baseweb="slider"] {
            color: #f63366 !important;
        }

        .stFileUploader {
            background-color: #1c1c1c;
            padding: 1em;
            border-radius: 8px;
            border: 1px solid #333;
        }

        .stDateInput > div > div {
            background-color: #1e1e1e;
            color: white;
            border-radius: 5px;
        }

        .css-1aumxhk {
            background-color: #262730;
            color: white;
            border-radius: 5px;
            padding: 0.4em 1em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Başlık kısmı
st.markdown("""
    <h1 style='color:#f5c518;'>⚡ WattGuardian</h1>
    <h3 style='color:white;'>Enerji Takibi ve Anomali Tespiti</h3>
    <hr style="border: 1px solid #444;">
""", unsafe_allow_html=True)


# 📥 Veri yükleme
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
    else:
        df = pd.read_csv("energydata_complete.csv", parse_dates=["date"])
    return df

# 🔴 Anomali hesaplama
@st.cache_data
def load_anomalies(df, threshold):
    mean = df['Appliances'].mean()
    std = df['Appliances'].std()
    df['z_score'] = (df['Appliances'] - mean) / std
    df['is_anomaly'] = df['z_score'].abs() > threshold
    return df[df['is_anomaly']]

# 🎯 Başlık
st.title("🔌 WattGuardian: Enerji Takibi ve Anomali Tespiti")

# 📁 Dosya yükleyici
uploaded_file = st.file_uploader("📁 Veri dosyası yükle (.csv, tarih sütunu 'date' olmalı)", type="csv")

# ⚙️ Threshold seçici
threshold = st.slider("Anomali Eşik Değeri (Z-Score)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)

# 📊 Veriyi yükle
df = load_data(uploaded_file)
anomalies = load_anomalies(df, threshold)

# Genel enerji tüketimi istatistikleri
mean_value = df['Appliances'].mean()
max_value = df['Appliances'].max()
min_value = df['Appliances'].min()
std_value = df['Appliances'].std()

# Üst satırda kutu kutu göster
st.markdown("### 🔍 Tüketim Verisi Genel Özeti")
col1, col2, col3, col4 = st.columns(4)

col1.metric("🔹 Ortalama", f"{mean_value:.2f} Wh")
col2.metric("🔺 Maksimum", f"{max_value} Wh")
col3.metric("🔻 Minimum", f"{min_value} Wh")
col4.metric("📊 Std Sapma", f"{std_value:.2f}")


# 📈 Anomali istatistikleri
anomaly_count = len(anomalies)
total_count = len(df)
anomaly_ratio = (anomaly_count / total_count) * 100
st.markdown(f"🔴 **{anomaly_count} adet anomali tespit edildi** (%{anomaly_ratio:.2f})")

# CSV olarak indirilecek dosyayı hazırla
csv = anomalies.to_csv(index=False).encode('utf-8')

# İndirme butonu
st.download_button(
    label="⬇️ Anomalileri CSV olarak indir",
    data=csv,
    file_name='anomalies.csv',
    mime='text/csv'
)


# 📆 Tarih aralığı seçimi
start_date = st.date_input("Başlangıç Tarihi", df['date'].min().date())
end_date = st.date_input("Bitiş Tarihi", df['date'].max().date())

if start_date > end_date:
    st.error("Başlangıç tarihi, bitiş tarihinden büyük olamaz!")
else:
    # 🧹 Filtrele
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]

    anomaly_mask = (anomalies['date'] >= pd.to_datetime(start_date)) & (anomalies['date'] <= pd.to_datetime(end_date))
    filtered_anomalies = anomalies.loc[anomaly_mask]

    # 📊 Grafik çizimi
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered_df['date'], filtered_df['Appliances'], label='Enerji Tüketimi', color='blue')

    if not filtered_anomalies.empty:
        ax.scatter(filtered_anomalies['date'], filtered_anomalies['Appliances'],
                   color='red', label='Anomaliler', s=50)

    ax.set_title("Enerji Tüketimi ve Anomaliler")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Tüketim (Wh)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
