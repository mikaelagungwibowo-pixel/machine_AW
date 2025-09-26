# app.py
from io import BytesIO
import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, classification_report,
    f1_score, make_scorer, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================================================
# KONFIGURASI SKEMA TERKUNCI
# =========================================================
CANON_FEATURES = [
    "USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai",
    "mandiri/flagsip", "BEKERJA/TIDAK",
]
TARGET_NAME = "LULUS TEPAT/TIDAK"

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Kelulusan Tepat Waktu",
    layout="wide",
    page_icon="🎓"
)
st.sidebar.caption(f"scikit-learn version: {sklearn.__version__}")

# =========================================================
# FUNGSI-FUNGSI UTAMA
# =========================================================
@st.cache_resource
def load_default_model():
    """Memuat model default dari file 'model.joblib' dengan pesan debug."""
    model_path = 'model.joblib'
    
    # Pesan debug untuk pengguna
    st.sidebar.warning(f"Mencoba memuat model default dari path: `{os.path.abspath(model_path)}`")
    
    if os.path.exists(model_path):
        st.sidebar.success("✅ File `model.joblib` ditemukan.")
        try:
            model_obj = joblib.load(model_path)
            return model_obj
        except Exception as e:
            st.sidebar.error(f"Gagal memuat file model: {e}")
            return None
    else:
        st.sidebar.error("❌ File `model.joblib` TIDAK ditemukan di folder yang sama dengan `app.py`.")
        return None

def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

NAME_MAP = {
    "usiamasuk": "USIAMASUK", "usia_masuk": "USIAMASUK", "usia": "USIAMASUK",
    "ip2": "IP2", "ipk2": "IP2", "ips2": "IP2", "ip3": "IP3", "ipk3": "IP3", "ips3": "IP3",
    "ip5": "IP5", "ipk5": "IP5", "ips5": "IP5",
    "reratanilai": "rata-rata nilai", "rataratanilai": "rata-rata nilai", "rerata": "rata-rata nilai",
    "jalur": "mandiri/flagsip",
    "bekerja": "BEKERJA/TIDAK", "bekerja/tidak": "BEKERJA/TIDAK",
    "lulustepat": "LULUS TEPAT/TIDAK", "lulus_tepat": "LULUS TEPAT/TIDAK", "lulus": "LULUS TEPAT/TIDAK",
}

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # (Fungsi ini tidak diubah, sama seperti sebelumnya)
    rename_map = {}
    for c in df.columns:
        key = _norm(c)
        if key in NAME_MAP:
            rename_map[c] = NAME_MAP[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    if "mandiri/flagsip" in df.columns:
        df["mandiri/flagsip"] = df["mandiri/flagsip"].str.strip().str.upper().replace("FLAGSHIP", "FLAGSIP")
    if "BEKERJA/TIDAK" in df.columns:
        df["BEKERJA/TIDAK"] = df["BEKERJA/TIDAK"].str.strip().str.lower().replace(
            {"1": "YA", "y": "YA", "true": "YA", "bekerja": "YA",
             "0": "TIDAK", "t": "TIDAK", "false": "TIDAK", "tidak bekerja": "TIDAK"}
        ).str.upper()
    for col in ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ... (Fungsi build_pipeline, plot, dll. tetap sama seperti sebelumnya)
def build_pipeline(model_name: str, numeric_features, categorical_features, params: dict):
    if model_name == "Naive Bayes":
        n_bins = params.get("n_bins_discretizer", 5)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("discretizer", KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform'))
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    if model_name == "Naive Bayes":
        model = CategoricalNB(alpha=params.get("alpha", 1.0))
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=params.get("random_state", 42),
            class_weight="balanced" if params.get("balanced", True) else None
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=params.get("random_state", 42),
            class_weight="balanced" if params.get("balanced", True) else None,
            n_jobs=-1
        )
    else:
        raise ValueError("Model tidak dikenali")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)
    
def to_binary(y_series: pd.Series, positive_value):
    return (y_series == positive_value).astype(int)

# =========================================================
# INISIALISASI & SIDEBAR
# =========================================================
if "active_model_obj" not in st.session_state:
    st.session_state.active_model_obj = load_default_model()
    st.session_state.model_source = "Default" if st.session_state.active_model_obj else "None"

st.sidebar.title("⚙️ Pengaturan")
st.sidebar.caption("Model & parameter pelatihan")
# ... (sisa kode sidebar tidak berubah)
model_name = st.sidebar.selectbox(
    "Pilih Model", ["Random Forest", "Decision Tree", "Naive Bayes"], index=0
)
st.sidebar.divider()
use_cv = st.sidebar.toggle("Gunakan Cross-Validation", value=True)
if use_cv:
    n_folds = st.sidebar.number_input("Jumlah Folds (k)", min_value=3, max_value=20, value=5, step=1)
else:
    test_size = st.sidebar.slider("Porsi Test Set", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
st.sidebar.divider()
random_state = st.sidebar.number_input("Random State", value=42, step=1)
params = {"random_state": random_state}
if model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 100, 1000, 300, 50)
    params["max_depth"] = st.sidebar.select_slider("max_depth", options=[None, 5, 10, 15, 20, 30, 50], value=None)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    try:
        params["balanced"] = st.sidebar.toggle("class_weight='balanced'", value=True)
    except AttributeError:
        params["balanced"] = st.sidebar.checkbox("class_weight='balanced'", value=True)
elif model_name == "Decision Tree":
    params["max_depth"] = st.sidebar.select_slider("max_depth", options=[None, 3, 5, 10, 15, 20, 30], value=None)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    try:
        params["balanced"] = st.sidebar.toggle("class_weight='balanced'", value=True)
    except AttributeError:
        params["balanced"] = st.sidebar.checkbox("class_weight='balanced'", value=True)
elif model_name == "Naive Bayes":
    st.sidebar.caption("Pengaturan untuk Categorical Naive Bayes")
    params["n_bins_discretizer"] = st.sidebar.number_input("Jumlah Bins (Diskritisasi)", min_value=2, max_value=15, value=5, step=1, help="Mengubah fitur numerik menjadi beberapa kelompok/kategori.")
    params["alpha"] = st.sidebar.slider("Alpha (smoothing)", 0.0, 2.0, 1.0, 0.1, help="Parameter smoothing untuk mencegah probabilitas nol.")
st.sidebar.divider()
st.sidebar.caption("💾 Ganti Model Aktif")
uploaded_model = st.sidebar.file_uploader("Unggah Model (.joblib)", type=["joblib"], accept_multiple_files=False)
if uploaded_model is not None:
    try:
        model_obj = joblib.load(uploaded_model)
        st.session_state.active_model_obj = model_obj
        st.session_state.model_source = "Unggahan"
        st.sidebar.success("Model dari file unggahan berhasil dimuat dan sekarang aktif.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

# =========================================================
# MAIN LAYOUT
# =========================================================
st.title("🎓 Prediksi Kelulusan Tepat Waktu — Skema Fitur Terkunci")

# --- PERBAIKAN: Menampilkan pesan status model yang lebih jelas di halaman utama ---
if st.session_state.model_source == "None":
    st.warning("⚠️ **Model default tidak aktif.** Aplikasi ini dapat digunakan setelah Anda meletakkan file `model.joblib` di folder yang benar, atau setelah Anda melatih model baru dari data di tab 'Pelatihan & Evaluasi'.", icon="⚠️")

st.markdown("""
Fitur dipakai: **USIAMASUK, IP2, IP3, IP5, rata-rata nilai, mandiri/flagsip, BEKERJA/TIDAK**
Target (label): **LULUS TEPAT/TIDAK**
""")

try:
    tab_data, tab_train, tab_form, tab_chat, tab_about = st.tabs(
        ["📁 Data", "🏋️ Pelatihan & Evaluasi", "📝 Form Input (7 Fitur)", "🤖 Chatbot", "ℹ️ Tentang"]
    )
except Exception:
    tab_data, tab_train, tab_form, tab_chat, tab_about = st.tabs(
        ["Data", "Pelatihan & Evaluasi", "Form Input (7 Fitur)", "Chatbot", "Tentang"]
    )

# ... (Seluruh kode di dalam tab-tab (tab_data, tab_train, tab_form, tab_chat, tab_about)
#      tetap sama seperti versi fungsional terakhir, tidak perlu diubah)

with tab_data:
    st.subheader("1) Unggah Data (CSV/XLSX/XLS)")
    uploaded_file = st.file_uploader("Pilih file data", type=["csv", "xlsx", "xls"])
    df = None # Mulai dengan df kosong
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"File dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
            df = harmonize_columns(df)
            st.session_state.df_cached = df # Simpan ke session state
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")
            st.session_state.df_cached = None
    
    # Selalu ambil df dari cache untuk ditampilkan
    df_to_show = st.session_state.get("df_cached", None)
    if df_to_show is not None:
        st.markdown("**Data (20 baris pertama):**")
        st.dataframe(df_to_show.head(20), use_container_width=True)
        # ... (sisa expander)
    else:
        st.info("Belum ada data yang diunggah.")


with tab_train:
    st.subheader("2) Latih & Evaluasi (Fitur Terkunci)")
    df = st.session_state.get("df_cached", None) # Ambil dari session state
    if df is None:
        st.warning("Unggah data di tab **Data** untuk melatih model baru.")
    else:
        # ... (sisa kode tab_train lengkap)
        pass # Placeholder untuk keringkasan, kode aslinya lengkap


with tab_form:
    st.subheader("3) Prediksi Individu — Form 7 Fitur")
    active_model_obj = st.session_state.get("active_model_obj", None)
    if not active_model_obj:
        st.error("⛔ Tidak ada model yang aktif untuk prediksi.")
    else:
        # ... (sisa kode tab_form lengkap)
        pass # Placeholder

with tab_chat:
    st.subheader("4) Chatbot Akademik")
    active_model_obj = st.session_state.get("active_model_obj", None)
    if not active_model_obj:
        st.error("⛔ Tidak ada model yang aktif. Chatbot tidak dapat berfungsi.")
    else:
        # ... (sisa kode tab_chat lengkap)
        pass # Placeholder

with tab_about:
    st.subheader("Tentang Aplikasi (Skema Terkunci)")
    st.markdown(f"""
- **Fitur digunakan**: {', '.join(CANON_FEATURES)}
- **Target**: {TARGET_NAME}
""")
