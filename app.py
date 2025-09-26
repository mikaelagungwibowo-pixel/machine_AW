# app.py
from io import BytesIO
import io
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
    "USIAMASUK",
    "IP2", "IP3", "IP5",
    "rata-rata nilai",
    "mandiri/flagsip",
    "BEKERJA/TIDAK",
]
TARGET_NAME = "LULUS TEPAT/TIDAK"

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Kelulusan Tepat Waktu — Skema Terkunci",
    layout="wide",
    page_icon="🎓"
)
st.sidebar.caption(f"scikit-learn version: {sklearn.__version__}")

# --- PERUBAHAN: Fungsi untuk memuat model default ---
@st.cache_resource
def load_default_model():
    """Memuat model default dari file 'model.joblib'."""
    try:
        model_path = 'model.joblib'
        model_obj = joblib.load(model_path)
        return model_obj
    except FileNotFoundError:
        return None

# =========================================================
# UTILITAS & FUNGSI LAINNYA
# =========================================================
# (Semua fungsi utilitas seperti harmonize_columns, build_pipeline, dll. tetap sama)
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

NAME_MAP = {
    "usiamasuk": "USIAMASUK", "usia_masuk": "USIAMASUK", "usiamasuktahun": "USIAMASUK",
    "usiamasukth": "USIAMASUK", "usia": "USIAMASUK", "usia masuk": "USIAMASUK",
    "ip2": "IP2", "ipk2": "IP2", "ips2": "IP2",
    "ip3": "IP3", "ipk3": "IP3", "ips3": "IP3",
    "ip5": "IP5", "ipk5": "IP5", "ips5": "IP5",
    "reratanilai": "rata-rata nilai", "rataratanilai": "rata-rata nilai", "rata2nilai": "rata-rata nilai",
    "rata-rata": "rata-rata nilai", "avgscore": "rata-rata nilai", "nilaiavg": "rata-rata nilai",
    "nilai_rerata": "rata-rata nilai", "rerata": "rata-rata nilai",
    "jalur": "mandiri/flagsip", "mandiri/flagsip": "mandiri/flagsip", "mandiriflagsip": "mandiri/flagsip",
    "mandiriflagship": "mandiri/flagsip",
    "bekerja": "BEKERJA/TIDAK", "bekerja/tidak": "BEKERJA/TIDAK", "statusbekerja": "BEKERJA/TIDAK",
    "lulustepat": "LULUS TEPAT/TIDAK", "lulustepattidak": "LULUS TEPAT/TIDAK",
    "lulus_tepat": "LULUS TEPAT/TIDAK", "lulus": "LULUS TEPAT/TIDAK",
    "statuslulus": "LULUS TEPAT/TIDAK", "lulustepat/tidak": "LULUS TEPAT/TIDAK",
}

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        key = _norm(c)
        if key in NAME_MAP:
            rename_map[c] = NAME_MAP[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    if "mandiri/flagsip" in df.columns:
        def norm_jalur(v):
            if pd.isna(v): return v
            s = str(v).strip().upper()
            if s == "FLAGSHIP": s = "FLAGSIP"
            return s
        df["mandiri/flagsip"] = df["mandiri/flagsip"].apply(norm_jalur)
    if "BEKERJA/TIDAK" in df.columns:
        def norm_bin_work(v):
            if pd.isna(v): return v
            s = str(v).strip().lower()
            if s in {"1", "ya", "y", "true", "bekerja"}: return "YA"
            if s in {"0", "tidak", "tdk", "t", "false", "tidak bekerja"}: return "TIDAK"
            return str(v).upper()
        df["BEKERJA/TIDAK"] = df["BEKERJA/TIDAK"].apply(norm_bin_work)
    for col in ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (df[col].astype(str).str.strip().str.replace(",", ".", regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def smart_detect_target(df: pd.DataFrame, target_guess: str = TARGET_NAME):
    if TARGET_NAME in df.columns: return TARGET_NAME
    candidates = [c for c in df.columns if _norm(c) in {"lulustepattidak", "lulus_tepat", "lulus", "statuslulus"}]
    return candidates[0] if candidates else df.columns[-1]

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

def plot_roc_pr(y_true_bin, y_score):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = roc_auc_score(y_true_bin, y_score)
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    st.pyplot(fig1)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(recall, precision)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    st.pyplot(fig2)

def to_binary(y_series: pd.Series, positive_value):
    return (y_series == positive_value).astype(int)

# =========================================================
# INISIALISASI MODEL & STATE APLIKASI
# =========================================================

# --- PERUBAHAN: Inisialisasi model aktif saat aplikasi dimulai ---
if "active_model_obj" not in st.session_state:
    default_model = load_default_model()
    if default_model:
        st.session_state.active_model_obj = default_model
        st.session_state.model_source = "Default"
    else:
        st.session_state.active_model_obj = None
        st.session_state.model_source = "None"
# --- AKHIR PERUBAHAN ---

df_cached = st.session_state.get("df_cached", None)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.caption("Model & parameter pelatihan")
model_name = st.sidebar.selectbox(
    "Pilih Model", ["Random Forest", "Decision Tree", "Naive Bayes"], index=0
)
st.sidebar.divider()
use_cv = st.sidebar.toggle("Gunakan Cross-Validation", value=True)
if use_cv:
    n_folds = st.sidebar.number_input("Jumlah Folds (k)", min_value=3, max_value=20, value=5, step=1)
    test_size = 0
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

# --- PERUBAHAN: Logika untuk mengganti model aktif ---
if uploaded_model is not None:
    try:
        model_obj = joblib.load(uploaded_model)
        st.session_state.active_model_obj = model_obj
        st.session_state.model_source = "Unggahan"
        st.sidebar.success("Model dari file unggahan berhasil dimuat dan sekarang aktif.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
# --- AKHIR PERUBAHAN ---

# =========================================================
# MAIN LAYOUT
# =========================================================
st.title("🎓 Prediksi Kelulusan Tepat Waktu — Skema Fitur Terkunci")
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

with tab_data:
    # ... kode tab data tidak berubah ...
    pass

with tab_train:
    st.subheader("2) Latih & Evaluasi (Fitur Terkunci)")
    if df is None:
        st.warning("Unggah data terlebih dahulu di tab **Data** untuk melatih model baru.")
    else:
        # ... (logika pelatihan tetap sama)
        if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
            # ... (semua logika pelatihan)
            
            # --- PERUBAHAN: Model hasil pelatihan menjadi model aktif ---
            st.session_state.active_model_obj = {
                "pipeline": pipe, 
                "features": locked_features, 
                "target": target_col, 
                "positive": positive_value
            }
            st.session_state.model_source = "Pelatihan"
            st.success("Model baru berhasil dilatih dan sekarang menjadi model aktif.")
            # --- AKHIR PERUBAHAN ---

# ... (sisa kode tab_train, chatbot utilities, dll. tetap sama)

# =========================================================
# KODE UNTUK TAB FORM, CHATBOT, ABOUT
# =========================================================

with tab_form:
    st.subheader("3) Prediksi Individu — Form 7 Fitur")
    active_model_obj = st.session_state.get("active_model_obj", None)
    
    # --- PERUBAHAN: Menampilkan status model yang aktif ---
    if active_model_obj:
        if st.session_state.model_source == "Default":
            st.info("ℹ️ Menggunakan **Model Default** yang sudah tersedia. Anda bisa melatih model baru di tab 'Pelatihan & Evaluasi'.")
        elif st.session_state.model_source == "Pelatihan":
            st.success("✅ Menggunakan **Model Hasil Pelatihan** yang baru saja Anda buat.")
        elif st.session_state.model_source == "Unggahan":
            st.success("✅ Menggunakan **Model dari File Unggahan**.")
    # --- AKHIR PERUBAHAN ---

    if active_model_obj is None:
        st.error("⛔ **Tidak ada model yang aktif.**\n\nSilakan letakkan file `model.joblib` di folder aplikasi, atau latih model baru di tab 'Pelatihan & Evaluasi'.")
    else:
        # (sisa logika tab_form tidak berubah)
        pass

with tab_chat:
    st.subheader("4) Chatbot Akademik — Tanya Jawab & Rekomendasi")
    active_model_obj = st.session_state.get("active_model_obj", None)
    
    # --- PERUBAHAN: Menampilkan status model yang aktif ---
    if active_model_obj:
        if st.session_state.model_source == "Default":
            st.info("ℹ️ Menggunakan **Model Default** yang sudah tersedia.")
        elif st.session_state.model_source == "Pelatihan":
            st.success("✅ Menggunakan **Model Hasil Pelatihan**.")
        elif st.session_state.model_source == "Unggahan":
            st.success("✅ Menggunakan **Model dari File Unggahan**.")
    # --- AKHIR PERUBAHAN ---

    if not active_model_obj:
        st.error("⛔ **Tidak ada model yang aktif.**\n\nChatbot tidak bisa berfungsi tanpa model. Silakan aktifkan model default atau latih model baru.")
    else:
        # (sisa logika tab_chat tidak berubah)
        pass

with tab_about:
    # ... kode tab about tidak berubah ...
    pass
