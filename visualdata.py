import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Dashboard Analisis Tarif Uber",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_clean_data():
    df = pd.read_csv('uber_clean.csv', nrows=50000)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df


@st.cache_data
def load_importance_data():
    return pd.read_csv('importance.csv')


@st.cache_data
def load_results_data():

    df = pd.read_csv('results.csv')

    # kalau format csv kebalik
    if 'Model' not in df.columns:

        df = df.transpose()
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index(drop=True)

    # ubah semua numerik
    for col in df.columns:
        if col != 'Model':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')


# =========================================================
# LOAD SEMUA DATA
# =========================================================

with st.spinner("Memuat sistem AI dan data historis..."):
    df_clean = load_clean_data()
    df_importance = load_importance_data()
    df_results = load_results_data()
    rf_model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Menu Navigasi")

halaman = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Kalkulator Prediksi",
        "Faktor Penentu Tarif",
        "Perbandingan Algoritma",
        "Pola Perjalanan",
        "Peta Kepadatan"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Dashboard ini menggunakan teknologi Machine Learning
untuk menganalisis pola tarif Uber berdasarkan data historis.
""")

# =========================================================
# HALAMAN 1
# =========================================================

if halaman == "Kalkulator Prediksi":

    st.title("Kalkulator Prediksi Tarif Uber Berbasis AI")

    st.markdown("""
    Dashboard ini memanfaatkan teknologi Machine Learning
    untuk melakukan prediksi tarif perjalanan Uber berdasarkan
    pola data historis.

    Sistem menggunakan algoritma Random Forest untuk mempelajari
    hubungan antara jarak perjalanan, waktu pemesanan,
    jumlah penumpang, dan variabel lainnya terhadap tarif perjalanan.

    Pengguna dapat melakukan simulasi prediksi tarif
    dengan memasukkan detail perjalanan pada form berikut.
    """)

    st.markdown("---")

    col_input, col_hasil = st.columns([2, 1])

    with col_input:

        st.subheader("Input Detail Perjalanan")

        col1, col2 = st.columns(2)

        with col1:

            input_distance = st.number_input(
                "Jarak Tempuh (KM)",
                min_value=0.5,
                max_value=100.0,
                value=5.0,
                step=0.5
            )

            input_hour = st.slider(
                "Jam Pemesanan",
                min_value=0,
                max_value=23,
                value=12
            )

            input_passengers = st.selectbox(
                "Jumlah Penumpang",
                [1, 2, 3, 4, 5, 6]
            )

        with col2:

            input_year = st.selectbox(
                "Tahun Acuan Tarif",
                [2009, 2010, 2011, 2012, 2013, 2014, 2015]
            )

            input_day = st.selectbox(
                "Hari",
                [
                    ("Senin", 0),
                    ("Selasa", 1),
                    ("Rabu", 2),
                    ("Kamis", 3),
                    ("Jumat", 4),
                    ("Sabtu", 5),
                    ("Minggu", 6)
                ],
                format_func=lambda x: x[0]
            )

            input_month = st.selectbox(
                "Bulan",
                [
                    ("Januari", 1),
                    ("Februari", 2),
                    ("Maret", 3),
                    ("April", 4),
                    ("Mei", 5),
                    ("Juni", 6),
                    ("Juli", 7),
                    ("Agustus", 8),
                    ("September", 9),
                    ("Oktober", 10),
                    ("November", 11),
                    ("Desember", 12)
                ],
                format_func=lambda x: x[0]
            )

        hitung_btn = st.button(
            "Hitung Estimasi Tarif",
            type="primary"
        )

    with col_hasil:
        
        st.subheader("Hasil Prediksi")

        if hitung_btn:

            input_data = pd.DataFrame({
                'distance_km': [input_distance],
                'passenger_count': [input_passengers],
                'hour': [input_hour],
                'day_of_week': [input_day[1]],
                'month': [input_month[1]],
                'year': [input_year]
            })

            predicted_fare = rf_model.predict(input_data)[0]

            st.success("Prediksi berhasil dilakukan!")

            fare_display = f"${predicted_fare:.2f}"

            html_content = f"""
<div style="padding:30px; border-radius:20px; background: linear-gradient(135deg, #1E3A8A, #2563EB); text-align:center; border:1px solid #3B82F6; box-shadow: 0 0 20px rgba(37,99,235,0.3);">
    <h3 style="color:white; margin-bottom:15px; font-size:24px;">Estimasi Tarif Perjalanan</h3>
    <h1 style="color:white; font-size:60px; margin:0; font-weight:bold;">{fare_display}</h1>
    <p style="color:#DBEAFE; margin-top:10px; font-size:18px;">USD</p>
</div>
            """

            st.markdown(html_content, unsafe_allow_html=True)

            st.info("""
            Estimasi tarif di atas merupakan hasil prediksi model Machine Learning
            berdasarkan data historis Uber.
            """)

        else:
            st.warning("Masukkan detail perjalanan terlebih dahulu.")

# =========================================================
# HALAMAN 2
# =========================================================

elif halaman == "Faktor Penentu Tarif":

    st.title("Analisis Faktor Penentu Tarif")

    st.markdown("""
    Halaman ini menampilkan hasil analisis variabel
    yang paling memengaruhi tarif Uber berdasarkan
    model Machine Learning Random Forest.
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rata-rata Tarif",
            f"${df_clean['fare_amount'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Rata-rata Jarak",
            f"{df_clean['distance_km'].mean():.1f} KM"
        )

    with col3:
        st.metric(
            "Jumlah Data",
            f"{len(df_clean):,}"
        )

    with col4:
        jam_sibuk = df_clean.groupby('hour').size().idxmax()

        st.metric(
            "Jam Sibuk",
            f"{jam_sibuk}:00"
        )

    st.markdown("---")

    st.subheader("Feature Importance")

    nama_fitur = {
        'distance_km': 'Jarak Tempuh',
        'year': 'Tahun',
        'hour': 'Jam',
        'month': 'Bulan',
        'day_of_week': 'Hari',
        'passenger_count': 'Jumlah Penumpang'
    }

    df_importance['Fitur_Indo'] = df_importance['Fitur'].map(nama_fitur)

    df_importance = df_importance.sort_values(
        by='Importance',
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x='Importance',
        y='Fitur_Indo',
        data=df_importance,
        palette='Blues_r',
        ax=ax
    )

    st.pyplot(fig)

    st.info("""
    Hasil analisis menunjukkan bahwa jarak perjalanan
    menjadi faktor paling dominan dalam menentukan tarif Uber.
    """)

# =========================================================
# HALAMAN 3
# =========================================================

elif halaman == "Perbandingan Algoritma":

    st.title("Perbandingan Algoritma Machine Learning")

    st.markdown("""
    Halaman ini digunakan untuk membandingkan performa
    beberapa algoritma Machine Learning dalam memprediksi
    tarif Uber berdasarkan data historis.

    Evaluasi dilakukan menggunakan beberapa metrik
    seperti R2 Score dan MAE untuk mengetahui
    model terbaik yang memiliki tingkat akurasi tinggi
    dan error yang rendah.
    """)

    st.markdown("---")

    st.subheader("Data Evaluasi")

    st.dataframe(df_results, use_container_width=True)

    st.markdown("---")

    # AUTO DETECT
    model_col = None
    r2_col = None
    mae_col = None

    for col in df_results.columns:

        lower = col.lower()

        if 'model' in lower:
            model_col = col

        if 'r2' in lower or 'r²' in lower:
            r2_col = col

        if 'mae' in lower:
            mae_col = col

    if model_col and r2_col and mae_col:

        best_r2 = df_results[r2_col].max()

        best_model = df_results.loc[
            df_results[r2_col].idxmax(),
            model_col
        ]

        lowest_mae = df_results[mae_col].min()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Jumlah Algoritma",
                len(df_results)
            )

        with col2:
            st.metric(
                "R2 Score Tertinggi",
                f"{best_r2:.3f}"
            )

        with col3:
            st.metric(
                "MAE Terendah",
                f"{lowest_mae:.3f}"
            )

        st.markdown("---")

        # GRAFIK R2
        st.subheader("Perbandingan R2 Score")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=model_col,
            y=r2_col,
            data=df_results,
            palette='Blues_r',
            ax=ax
        )

        st.pyplot(fig)

        st.markdown("---")

        # GRAFIK MAE
        st.subheader("Perbandingan MAE")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=model_col,
            y=mae_col,
            data=df_results,
            palette='Reds_r',
            ax=ax
        )

        st.pyplot(fig)

        st.success(f"""
        Model terbaik adalah {best_model}
        dengan nilai R2 Score sebesar {best_r2:.3f}
        """)

    else:

        st.error("Format results.csv masih belum sesuai.")

# =========================================================
# HALAMAN 4
# =========================================================

elif halaman == "Pola Perjalanan":

    st.title("Eksplorasi Pola Data Historis")

    st.markdown("""
    Visualisasi berikut digunakan untuk memahami
    hubungan antara jarak perjalanan,
    waktu pemesanan, dan tarif Uber.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribusi Tarif Berdasarkan Jarak")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            x='distance_km',
            y='fare_amount',
            data=df_clean.sample(2000),
            alpha=0.3,
            color='red',
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Fluktuasi Tarif Harian")

        hourly_fare = df_clean.groupby(
            'hour'
        )['fare_amount'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.lineplot(
            x='hour',
            y='fare_amount',
            data=hourly_fare,
            marker='o',
            ax=ax
        )

        st.pyplot(fig)

# =========================================================
# HALAMAN 5
# =========================================================

elif halaman == "Peta Kepadatan":

    st.title("Peta Kepadatan Penjemputan Uber")

    st.markdown("""
    Visualisasi peta berikut menunjukkan
    persebaran lokasi penjemputan Uber
    berdasarkan data historis perjalanan.
    """)

    st.markdown("---")

    map_data = df_clean[
        ['pickup_latitude', 'pickup_longitude']
    ].rename(columns={
        'pickup_latitude': 'lat',
        'pickup_longitude': 'lon'
    })

    st.map(map_data.head(5000))

    st.info("""
    Area dengan titik paling padat menunjukkan
    wilayah dengan aktivitas pemesanan Uber tertinggi.
    """)