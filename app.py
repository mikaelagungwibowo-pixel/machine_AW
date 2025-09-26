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
# --- PERUBAHAN: Impor KBinsDiscretizer dan CategoricalNB ---
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB, CategoricalNB
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

# =========================================================
# UTILITAS & HARMONISASI KOLUMN
# =========================================================
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

# (Fungsi harmonize_columns dan smart_detect_target tetap sama)
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

# --- PERUBAHAN: Pipeline build_pipeline yang dimodifikasi ---
def build_pipeline(model_name: str, numeric_features, categorical_features, params: dict):
    # Logika pipeline numerik sekarang kondisional
    if model_name == "Naive Bayes":
        # Untuk Naive Bayes, kita lakukan diskritisasi (binning)
        n_bins = params.get("n_bins_discretizer", 5)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("discretizer", KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform'))
        ])
    else:
        # Untuk model lain, kita lakukan scaling seperti biasa
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

    # Logika pemilihan model sekarang menggunakan CategoricalNB
    if model_name == "Naive Bayes":
        # Gunakan CategoricalNB karena semua fitur sekarang bersifat diskrit
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

# (Fungsi-fungsi plot dan utilitas lainnya tetap sama)
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
# SIDEBAR PARAMETER
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
# --- PERUBAHAN: Pengaturan sidebar untuk Naive Bayes yang baru ---
elif model_name == "Naive Bayes":
    st.sidebar.caption("Pengaturan untuk Categorical Naive Bayes")
    params["n_bins_discretizer"] = st.sidebar.number_input("Jumlah Bins (Diskritisasi)", min_value=2, max_value=15, value=5, step=1, help="Mengubah fitur numerik menjadi beberapa kelompok/kategori.")
    params["alpha"] = st.sidebar.slider("Alpha (smoothing)", 0.0, 2.0, 1.0, 0.1, help="Parameter smoothing untuk mencegah probabilitas nol. Mirip seperti var_smoothing.")
# --- AKHIR PERUBAHAN ---

st.sidebar.divider()
st.sidebar.caption("💾 Muat model .joblib (opsional)")
uploaded_model = st.sidebar.file_uploader("Muat Model (.joblib)", type=["joblib"], accept_multiple_files=False)

# =========================================================
# MAIN LAYOUT dan TAB-TAB LAINNYA
# =========================================================
# (Tidak ada perubahan pada kode di bawah ini, 
#  semua logika Tab Data, Pelatihan, Form, Chatbot, dll. tetap sama)
df_cached = st.session_state.get("df_cached", None)

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
    # (Kode di dalam tab_data tidak berubah)
    st.subheader("1) Unggah Data (CSV/XLSX/XLS)")
    uploaded_file = st.file_uploader("Pilih file data", type=["csv", "xlsx", "xls"])
    df = None
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.success(f"CSV dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
            else:
                data_bytes = uploaded_file.read()
                xls = pd.ExcelFile(BytesIO(data_bytes))
                sheet = st.selectbox("Pilih sheet", xls.sheet_names, index=0)
                df = pd.read_excel(BytesIO(data_bytes), sheet_name=sheet)
                st.success(f"Excel dimuat: sheet '{sheet}' — {df.shape[0]} baris, {df.shape[1]} kolom.")
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")
            df = None
    else:
        df = df_cached

    if df is None:
        st.info("Belum ada file diunggah. Anda bisa mengunduh Template Excel di bawah.")

    if df is not None:
        df = harmonize_columns(df)
        st.session_state["df_cached"] = df
        st.markdown("**Data (20 baris pertama):**")
        st.dataframe(df.head(20), use_container_width=True)

        with st.expander("🔎 Pemeriksaan Skema"):
            must_have = CANON_FEATURES + [TARGET_NAME]
            present = [c for c in must_have if c in df.columns]
            missing = [c for c in must_have if c not in df.columns]
            st.write("Kolom ditemukan:", present)
            if missing:
                st.warning(f"Kolom belum ada: {missing} — pelatihan tetap bisa dengan fitur yang tersedia (target tetap wajib).")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tipe Data**")
                st.write(df.dtypes)
            with col2:
                st.markdown("**Missing Value per Kolom**")
                st.write(df.isna().sum())

        st.divider()
        st.markdown("**⬇️ Unduh Template Excel (header sesuai skema)**")
        sample_rows = [
            {"USIAMASUK": 18, "IP2": 3.2, "IP3": 3.3, "IP5": 3.4, "rata-rata nilai": 82, "mandiri/flagsip": "MANDIRI", "BEKERJA/TIDAK": "TIDAK", "LULUS TEPAT/TIDAK": "TEPAT"},
            {"USIAMASUK": 19, "IP2": 3.0, "IP3": 3.0, "IP5": 2.9, "rata-rata nilai": 75, "mandiri/flagsip": "FLAGSIP", "BEKERJA/TIDAK": "YA", "LULUS TEPAT/TIDAK": "TIDAK"},
        ]
        tpl_df = pd.DataFrame(sample_rows)
        excel_buf = BytesIO()
        try:
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                tpl_df.to_excel(writer, index=False, sheet_name="DATA")
            st.download_button(
                "Unduh Template Excel (sample_data_skematerkunci.xlsx)",
                data=excel_buf.getvalue(),
                file_name="sample_data_skematerkunci.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            csv_bytes = tpl_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Template CSV (sample_data_skematerkunci.csv)",
                data=csv_bytes,
                file_name="sample_data_skematerkunci.csv",
                mime="text/csv"
            )

with tab_train:
    # (Kode di dalam tab_train tidak berubah, karena sudah menggunakan build_pipeline yang baru)
    st.subheader("2) Latih & Evaluasi (Fitur Terkunci)")
    df = st.session_state.get("df_cached", None)
    if df is None:
        st.warning("Unggah data terlebih dahulu di tab **Data**.")
    else:
        target_col = smart_detect_target(df, TARGET_NAME)
        if target_col not in df.columns:
            st.error(f"Target **{TARGET_NAME}** tidak ditemukan. Pastikan ada kolom target.")
        else:
            locked_features = [c for c in CANON_FEATURES if c in df.columns]
            if not locked_features:
                st.error("Tidak ada satu pun fitur terkunci yang ditemukan di dataset.")
            else:
                st.success(f"Fitur dipakai: {locked_features}")
                numeric_features = [c for c in ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai"] if c in locked_features]
                categorical_features = [c for c in ["mandiri/flagsip", "BEKERJA/TIDAK"] if c in locked_features]

                unique_target_vals = sorted(df[target_col].dropna().unique().tolist(), key=lambda x: str(x))
                default_positive = (
                    "TEPAT" if "TEPAT" in unique_target_vals
                    else ("YA" if "YA" in unique_target_vals
                          else (1 if 1 in unique_target_vals else unique_target_vals[0]))
                )
                positive_value = st.selectbox(
                    "Nilai target yang dianggap **positif** (Kelulusan Tepat Waktu)",
                    options=unique_target_vals,
                    index=unique_target_vals.index(default_positive) if default_positive in unique_target_vals else 0
                )

                run_train = st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True)
                if run_train:
                    X = df[locked_features].copy()
                    y = df[target_col].copy()
                    
                    pipe = build_pipeline(model_name, numeric_features, categorical_features, params)

                    if use_cv:
                        st.info(f"Cross-Validation Aktif ({n_folds} folds)")
                        
                        scoring_metrics = {
                            'accuracy': 'accuracy',
                            'precision': make_scorer(precision_score, pos_label=positive_value, average='binary', zero_division=0),
                            'recall': make_scorer(recall_score, pos_label=positive_value, average='binary', zero_division=0),
                            'f1': make_scorer(f1_score, pos_label=positive_value, average='binary', zero_division=0)
                        }

                        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                        
                        scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring_metrics, n_jobs=-1)
                        
                        st.subheader(f"Hasil Rata-rata Cross-Validation ({n_folds} folds)")
                        results_df = pd.DataFrame({
                            "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
                            "Rata-rata": [scores['test_accuracy'].mean(), scores['test_precision'].mean(), scores['test_recall'].mean(), scores['test_f1'].mean()],
                            "Standar Deviasi": [scores['test_accuracy'].std(), scores['test_precision'].std(), scores['test_recall'].std(), scores['test_f1'].std()]
                        })
                        st.dataframe(results_df.style.format({
                            "Rata-rata": "{:.3f}",
                            "Standar Deviasi": "±{:.3f}"
                        }), use_container_width=True)

                        st.info("Setelah evaluasi, model final dilatih pada seluruh dataset untuk diunduh.")
                        pipe.fit(X, y)
                        
                    else: 
                        st.info("Mode Pelatihan: Train-Test Split")
                        y_bin = to_binary(y, positive_value)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state,
                            stratify=y_bin if len(np.unique(y_bin)) == 2 else None
                        )
                        y_test_bin = to_binary(y_test, positive_value)

                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average="binary", pos_label=positive_value, zero_division=0
                        )
                        st.success(
                            f"**Evaluasi (Test Set)** — Accuracy: **{acc:.3f}** "
                            f"Precision: **{prec:.3f}** "
                            f"Recall: **{rec:.3f}** "
                            f"F1: **{f1:.3f}**"
                        )

                        st.text("Classification Report:")
                        st.code(classification_report(y_test, y_pred, zero_division=0))

                        labels_for_cm = list(dict.fromkeys([positive_value] + [v for v in unique_target_vals if v != positive_value]))
                        cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm[:2] if len(labels_for_cm) >= 2 else labels_for_cm)
                        st.markdown("**Confusion Matrix**")
                        try:
                            plot_labels = labels_for_cm[:2] if cm.shape == (2, 2) else labels_for_cm[: cm.shape[0]]
                            plot_confusion_matrix(cm, labels=[str(l) for l in plot_labels])
                        except Exception:
                            st.write(cm)
                        
                        y_score = None
                        if hasattr(pipe.named_steps["model"], "predict_proba"):
                            try:
                                proba = pipe.predict_proba(X_test)
                                classes = pipe.named_steps["model"].classes_
                                pos_index = list(classes).index(positive_value) if positive_value in classes else (1 if proba.shape[1] > 1 else 0)
                                y_score = proba[:, pos_index]
                            except Exception:
                                y_score = None

                        if y_score is not None and len(np.unique(y_test_bin)) == 2:
                            st.markdown("**ROC & PR Curve**")
                            plot_roc_pr(y_test_bin, y_score)
                        else:
                            st.info("ROC/PR tidak tersedia (model tidak menyediakan probabilitas atau target tidak biner).")
                    
                    st.markdown("**Pentingnya Fitur**")
                    try:
                        if use_cv:
                            _, X_test_perm, _, y_test_perm = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
                        else:
                            X_test_perm, y_test_perm = X_test, y_test

                        model = pipe.named_steps["model"]
                        pre = pipe.named_steps["preprocess"]
                        
                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                            feature_names = pre.get_feature_names_out()
                            imp_df = pd.DataFrame({"fitur": feature_names, "importance": importances})
                        else:
                            scorer = make_scorer(f1_score, pos_label=positive_value, zero_division=0)
                            result = permutation_importance(
                                pipe, X_test_perm, y_test_perm,
                                n_repeats=5, random_state=random_state, n_jobs=-1, scoring=scorer
                            )
                            importances = result.importances_mean
                            feature_names = X_test_perm.columns.tolist()
                            imp_df = pd.DataFrame({"fitur": feature_names, "importance": importances})

                        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.barplot(data=imp_df, y="fitur", x="importance", ax=ax, color="#4C78A8")
                        ax.set_title("20 Fitur Teratas")
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Gagal menghitung importance: {e}")

                    st.markdown("**💾 Simpan Model**")
                    buf = io.BytesIO()
                    joblib.dump({"pipeline": pipe, "features": locked_features, "target": target_col, "positive": positive_value}, buf)
                    buf.seek(0)
                    st.download_button(
                        "Unduh Model (.joblib)",
                        data=buf.getvalue(),
                        file_name=f"model_{model_name.replace(' ', '_').lower()}_skema_terkunci.joblib",
                        mime="application/octet-stream"
                    )

                    st.session_state["last_trained_model"] = {
                        "pipeline": pipe,
                        "features": locked_features,
                        "target": target_col,
                        "positive": positive_value
                    }

# ... (Sisa kode untuk tab lain tetap sama)
# Anda bisa copy-paste bagian ini dari kode sebelumnya jika perlu
with tab_chat:
    pass # Letakkan kode tab_chat di sini
with tab_form:
    pass # Letakkan kode tab_form di sini
with tab_about:
    pass # Letakkan kode tab_about di sini
