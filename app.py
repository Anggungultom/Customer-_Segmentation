"""
Aplikasi Segmentasi Pelanggan Menggunakan Algoritma K-Means Clustering
Project Akhir Praktikum Data Mining 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Segmentasi Pelanggan - K-Means Clustering",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 Segmentasi Pelanggan Menggunakan K-Means Clustering")
st.markdown("""
Aplikasi ini mengelompokkan pelanggan ke dalam beberapa segmen menggunakan
algoritma **K-Means Clustering** berdasarkan fitur-fitur numerik yang kamu pilih
(misalnya usia, pendapatan, lokasi geografis, dll — sesuaikan dengan dataset kamu).
""")

# =========================================================
# SIDEBAR - UPLOAD DATASET
# =========================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file CSV dataset pelanggan", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("""
**Catatan:**
Pilih kolom numerik yang relevan untuk clustering
(hindari kolom seperti ID, nama, email, nomor telepon).

Contoh kolom yang bagus untuk clustering:
- Age
- Latitude / Longitude
- Annual Income
- Spending Score
""")

# =========================================================
# FUNGSI BANTUAN
# =========================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

def preprocess_data(df, feature_cols):
    data = df[feature_cols].copy()
    data = data.dropna()

    encoders = {}
    for col in feature_cols:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    return data, encoders

# =========================================================
# MAIN APP
# =========================================================
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("1️⃣ Data Preview")
    st.dataframe(df.head(10))
    st.write(f"Jumlah baris: **{df.shape[0]:,}**, Jumlah kolom: **{df.shape[1]}**")

    st.subheader("2️⃣ Statistik Deskriptif")
    st.dataframe(df.describe(include="all").transpose())

    # -----------------------------------------------------
    # PILIH KOLOM FITUR
    # -----------------------------------------------------
    st.subheader("3️⃣ Pilih Fitur untuk Clustering")
    all_columns = df.columns.tolist()

    # Tebak kolom numerik sebagai default
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else all_columns[:2]

    feature_cols = st.multiselect(
        "Pilih minimal 2 kolom fitur (disarankan kolom numerik seperti Age, Income, Latitude, dll)",
        all_columns,
        default=default_cols
    )

    if len(feature_cols) < 2:
        st.warning("Silakan pilih minimal 2 kolom fitur untuk melakukan clustering.")
        st.stop()

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    data, encoders = preprocess_data(df, feature_cols)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # -----------------------------------------------------
    # VISUALISASI EKSPLORATIF
    # -----------------------------------------------------
    st.subheader("4️⃣ Visualisasi Data Sebelum Clustering")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Distribusi {feature_cols[0]}**")
        fig, ax = plt.subplots()
        sns.histplot(data[feature_cols[0]], kde=True, ax=ax, color="steelblue")
        st.pyplot(fig)

    with col2:
        st.markdown("**Korelasi Antar Fitur**")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap="Blues", ax=ax, fmt=".2f")
        st.pyplot(fig)

    # -----------------------------------------------------
    # MENENTUKAN JUMLAH CLUSTER OPTIMAL (ELBOW METHOD)
    # -----------------------------------------------------
    st.subheader("5️⃣ Menentukan Jumlah Cluster Optimal (Elbow Method)")

    max_k = st.slider("Maksimal jumlah cluster untuk dicoba", 2, 15, 10)

    if st.button("📈 Hitung Elbow Method"):
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(data_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(data_scaled, labels))

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(list(k_range), inertias, marker="o", color="steelblue")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.plot(list(k_range), silhouettes, marker="o", color="orange")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score per k")
            st.pyplot(fig)

        st.info("""
        **Cara membaca:** Pilih titik di grafik Elbow Method di mana penurunan
        inertia mulai melandai (membentuk 'siku'). Silhouette Score lebih tinggi
        menunjukkan kualitas clustering yang lebih baik (mendekati 1).
        """)

    # -----------------------------------------------------
    # TRAINING MODEL K-MEANS
    # -----------------------------------------------------
    st.subheader("6️⃣ Jalankan K-Means Clustering")

    n_clusters = st.number_input("Jumlah cluster (k)", min_value=2, max_value=15, value=3, step=1)

    if st.button("🚀 Latih Model K-Means"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)

        result_df = df.loc[data.index].copy()
        result_df["Cluster"] = cluster_labels

        st.session_state["result_df"] = result_df
        st.session_state["feature_cols"] = feature_cols
        st.session_state["data_scaled"] = data_scaled
        st.session_state["cluster_labels"] = cluster_labels
        st.session_state["kmeans"] = kmeans

        sil_score = silhouette_score(data_scaled, cluster_labels)

        st.success("Clustering berhasil dijalankan!")
        st.metric("Silhouette Score", f"{sil_score:.4f}")

        # -----------------------------------------------------
        # HASIL CLUSTER
        # -----------------------------------------------------
        st.markdown("**Jumlah Anggota per Cluster**")
        cluster_counts = result_df["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.markdown("**Preview Data dengan Label Cluster**")
        st.dataframe(result_df.head(20))

        # Statistik rata-rata tiap cluster
        st.markdown("**Rata-rata Fitur per Cluster**")
        numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_feature_cols:
            cluster_summary = result_df.groupby("Cluster")[numeric_feature_cols].mean()
            st.dataframe(cluster_summary)

        # -----------------------------------------------------
        # VISUALISASI CLUSTER
        # -----------------------------------------------------
        st.markdown("**Visualisasi Cluster**")

        if len(feature_cols) == 2:
            # Langsung plot 2D
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                data[feature_cols[0]], data[feature_cols[1]],
                c=cluster_labels, cmap="tab10", alpha=0.6
            )
            centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(centers_original[:, 0], centers_original[:, 1],
                       c="red", marker="X", s=200, label="Centroid")
            ax.set_xlabel(feature_cols[0])
            ax.set_ylabel(feature_cols[1])
            ax.legend()
            st.pyplot(fig)
        else:
            # Reduksi dimensi pakai PCA kalau fitur > 2
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                data_pca[:, 0], data_pca[:, 1],
                c=cluster_labels, cmap="tab10", alpha=0.6
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Visualisasi Cluster (Reduksi Dimensi dengan PCA)")
            st.pyplot(fig)
            st.caption(f"Total variance explained oleh 2 komponen PCA: {sum(pca.explained_variance_ratio_):.2%}")

    # -----------------------------------------------------
    # DOWNLOAD HASIL
    # -----------------------------------------------------
    if "result_df" in st.session_state:
        st.subheader("7️⃣ Download Hasil Clustering")
        csv = st.session_state["result_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Hasil Clustering",
            data=csv,
            file_name="hasil_segmentasi_pelanggan.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Silakan upload dataset CSV terlebih dahulu melalui sidebar untuk memulai.")
    st.markdown("""
    ### Dataset yang Digunakan
    [Customer Segmentation (51k Records) - Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-segmentation-51k-records)

    Dataset ini berisi data demografis pelanggan (nama, gender, kota, negara,
    latitude/longitude, dll). Untuk clustering, pilih kolom numerik yang relevan
    seperti **Age**, **Latitude**, **Longitude**, atau kolom numerik lain yang tersedia.

    Hindari memilih kolom seperti ID, nama, email, atau nomor telepon karena
    tidak bermakna secara statistik untuk clustering.
    """)
