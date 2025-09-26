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
st.sidebar.caption("💾 Muat model .joblib (opsional)")
uploaded_model = st.sidebar.file_uploader("Muat Model (.joblib)", type=["joblib"], accept_multiple_files=False)

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
    st.subheader("1) Unggah Data (CSV/XLSX/XLS)")
    uploaded_file = st.file_uploader("Pilih file data", type=["csv", "xlsx", "xls"])
    df = None
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"File dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")
            df = None
    else:
        df = df_cached

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
                st.warning(f"Kolom belum ada: {missing}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tipe Data**")
                st.write(df.dtypes)
            with col2:
                st.markdown("**Missing Value per Kolom**")
                st.write(df.isna().sum())
    else:
        st.info("Belum ada file diunggah.")

with tab_train:
    st.subheader("2) Latih & Evaluasi (Fitur Terkunci)")
    if df is None:
        st.warning("Unggah data terlebih dahulu di tab **Data**.")
    else:
        target_col = smart_detect_target(df, TARGET_NAME)
        if target_col not in df.columns:
            st.error(f"Target **{TARGET_NAME}** tidak ditemukan.")
        else:
            locked_features = [c for c in CANON_FEATURES if c in df.columns]
            if not locked_features:
                st.error("Tidak ada fitur terkunci yang ditemukan.")
            else:
                st.success(f"Fitur dipakai: {locked_features}")
                numeric_features = [c for c in CANON_FEATURES if df[c].dtype in ['int64', 'float64'] and c in locked_features]
                categorical_features = [c for c in CANON_FEATURES if df[c].dtype == 'object' and c in locked_features]
                unique_target_vals = sorted(df[target_col].dropna().unique().tolist())
                default_positive = "TEPAT" if "TEPAT" in unique_target_vals else unique_target_vals[0]
                positive_value = st.selectbox("Nilai target positif", options=unique_target_vals, index=unique_target_vals.index(default_positive))

                if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
                    X = df[locked_features]
                    y = df[target_col]
                    pipe = build_pipeline(model_name, numeric_features, categorical_features, params)
                    if use_cv:
                        st.info(f"Cross-Validation Aktif ({n_folds} folds)")
                        scoring_metrics = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score, pos_label=positive_value, average='binary', zero_division=0), 'recall': make_scorer(recall_score, pos_label=positive_value, average='binary', zero_division=0), 'f1': make_scorer(f1_score, pos_label=positive_value, average='binary', zero_division=0)}
                        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                        scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring_metrics, n_jobs=-1)
                        st.subheader(f"Hasil Rata-rata Cross-Validation ({n_folds} folds)")
                        results_df = pd.DataFrame({"Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"], "Rata-rata": [scores['test_accuracy'].mean(), scores['test_precision'].mean(), scores['test_recall'].mean(), scores['test_f1'].mean()], "Standar Deviasi": [scores['test_accuracy'].std(), scores['test_precision'].std(), scores['test_recall'].std(), scores['test_f1'].std()]})
                        st.dataframe(results_df.style.format({"Rata-rata": "{:.3f}", "Standar Deviasi": "±{:.3f}"}), use_container_width=True)
                        st.info("Model final dilatih pada seluruh dataset.")
                        pipe.fit(X, y)
                    else:
                        st.info("Mode Pelatihan: Train-Test Split")
                        y_bin = to_binary(y, positive_value)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_bin if len(np.unique(y_bin)) == 2 else None)
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=positive_value, zero_division=0)
                        st.success(f"**Evaluasi (Test Set)** — Accuracy: **{acc:.3f}** Precision: **{prec:.3f}** Recall: **{rec:.3f}** F1: **{f1:.3f}**")
                        st.text("Classification Report:")
                        st.code(classification_report(y_test, y_pred, zero_division=0))
                        # Visualizations for train/test split
                        y_test_bin = to_binary(y_test, positive_value)
                        cm = confusion_matrix(y_test, y_pred)
                        st.markdown("**Confusion Matrix**")
                        plot_confusion_matrix(cm, labels=pipe.classes_)
                        if hasattr(pipe, "predict_proba"):
                            y_score = pipe.predict_proba(X_test)[:, 1]
                            st.markdown("**ROC & PR Curve**")
                            plot_roc_pr(y_test_bin, y_score)
                    
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
                            result = permutation_importance(pipe, X_test_perm, y_test_perm, n_repeats=5, random_state=random_state, n_jobs=-1, scoring=scorer)
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
                    st.download_button("Unduh Model (.joblib)", data=buf.getvalue(), file_name=f"model.joblib", mime="application/octet-stream")
                    st.session_state["last_trained_model"] = {"pipeline": pipe, "features": locked_features, "target": target_col, "positive": positive_value}

# --- KODE UNTUK TAB LAIN DIKEMBALIKAN ---
with tab_form:
    st.subheader("3) Prediksi Individu — Form 7 Fitur")
    active_model_obj = st.session_state.get("last_trained_model")
    if not active_model_obj:
        st.warning("Latih model di tab 'Pelatihan & Evaluasi' terlebih dahulu.")
    else:
        pipe = active_model_obj["pipeline"]
        positive_value = active_model_obj["positive"]
        with st.form("form7"):
            colA, colB, colC = st.columns(3)
            with colA:
                USIAMASUK = st.number_input("USIAMASUK (tahun)", 15, 60, 19)
                IP2 = st.number_input("IP2", 0.0, 4.0, 3.2, 0.01, "%.2f")
            with colB:
                IP3 = st.number_input("IP3", 0.0, 4.0, 3.2, 0.01, "%.2f")
                IP5 = st.number_input("IP5", 0.0, 4.0, 3.2, 0.01, "%.2f")
            with colC:
                rata_rata = st.slider("rata-rata nilai", 0, 100, 82)
                jalur = st.selectbox("mandiri/flagsip", ["MANDIRI", "FLAGSIP"])
                bekerja = st.selectbox("BEKERJA/TIDAK", ["YA", "TIDAK"])
            
            if st.form_submit_button("🔮 Prediksi"):
                inputs_form = {"USIAMASUK": USIAMASUK, "IP2": IP2, "IP3": IP3, "IP5": IP5, "rata-rata nilai": rata_rata, "mandiri/flagsip": jalur, "BEKERJA/TIDAK": bekerja}
                X_one = pd.DataFrame([inputs_form])
                pred = pipe.predict(X_one)[0]
                proba_str = ""
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_one)
                    pos_index = list(pipe.classes_).index(positive_value)
                    p_pos = proba[0, pos_index]
                    proba_str = f" — Probabilitas({positive_value}): **{p_pos:.3f}**"
                st.success(f"**Hasil Prediksi:** **{pred}**{proba_str}")

with tab_chat:
    st.subheader("4) Chatbot Akademik")
    st.info("Fitur chatbot sedang dalam pengembangan.")

with tab_about:
    st.subheader("Tentang Aplikasi (Skema Terkunci)")
    st.markdown(f"""
- **Fitur digunakan**: {', '.join(CANON_FEATURES)}
- **Target**: {TARGET_NAME}
- **Catatan**: Aplikasi ini menggunakan skema fitur yang telah ditentukan untuk memprediksi kelulusan tepat waktu.
""")
