# app.py
from io import BytesIO
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn  # tampilkan versi di sidebar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, classification_report,
    f1_score, make_scorer, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
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

# Pemetaan variasi nama kolom → nama kanonik
NAME_MAP = {
    # USIAMASUK
    "usiamasuk": "USIAMASUK",
    "usia_masuk": "USIAMASUK",
    "usiamasuktahun": "USIAMASUK",
    "usiamasukth": "USIAMASUK",
    "usia": "USIAMASUK",
    "usia masuk": "USIAMASUK",
    # IP series
    "ip2": "IP2", "ipk2": "IP2", "ips2": "IP2",
    "ip3": "IP3", "ipk3": "IP3", "ips3": "IP3",
    "ip5": "IP5", "ipk5": "IP5", "ips5": "IP5",
    # Rata-rata nilai
    "reratanilai": "rata-rata nilai",
    "rataratanilai": "rata-rata nilai",
    "rata2nilai": "rata-rata nilai",
    "rata-rata": "rata-rata nilai",
    "avgscore": "rata-rata nilai",
    "nilaiavg": "rata-rata nilai",
    "nilai_rerata": "rata-rata nilai",
    "rerata": "rata-rata nilai",
    # Jalur
    "jalur": "mandiri/flagsip",
    "mandiri/flagsip": "mandiri/flagsip",
    "mandiriflagsip": "mandiri/flagsip",
    "mandiriflagship": "mandiri/flagsip",
    # Bekerja
    "bekerja": "BEKERJA/TIDAK",
    "bekerja/tidak": "BEKERJA/TIDAK",
    "statusbekerja": "BEKERJA/TIDAK",
    # Target
    "lulustepat": "LULUS TEPAT/TIDAK",
    "lulustepattidak": "LULUS TEPAT/TIDAK",
    "lulus_tepat": "LULUS TEPAT/TIDAK",
    "lulus": "LULUS TEPAT/TIDAK",
    "statuslulus": "LULUS TEPAT/TIDAK",
    "lulustepat/tidak": "LULUS TEPAT/TIDAK",
}

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Rename ke kanonik
    rename_map = {}
    for c in df.columns:
        key = _norm(c)
        if key in NAME_MAP:
            rename_map[c] = NAME_MAP[key]
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2) Normalisasi nilai kategorikal
    if "mandiri/flagsip" in df.columns:
        def norm_jalur(v):
            if pd.isna(v):
                return v
            s = str(v).strip().upper()
            if s == "FLAGSHIP":
                s = "FLAGSIP"  # ejaan yang dipakai di skema
            return s
        df["mandiri/flagsip"] = df["mandiri/flagsip"].apply(norm_jalur)

    if "BEKERJA/TIDAK" in df.columns:
        def norm_bin_work(v):
            if pd.isna(v):
                return v
            s = str(v).strip().lower()
            if s in {"1", "ya", "y", "true", "bekerja"}:
                return "YA"
            if s in {"0", "tidak", "tdk", "t", "false", "tidak bekerja"}:
                return "TIDAK"
            return str(v).upper()
        df["BEKERJA/TIDAK"] = df["BEKERJA/TIDAK"].apply(norm_bin_work)

    # 3) Pastikan numerik benar-benar numerik (normalisasi koma → titik terlebih dahulu)
    for col in ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(",", ".", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def smart_detect_target(df: pd.DataFrame, target_guess: str = TARGET_NAME):
    if TARGET_NAME in df.columns:
        return TARGET_NAME
    candidates = [c for c in df.columns if _norm(c) in {"lulustepattidak", "lulus_tepat", "lulus", "statuslulus"}]
    return candidates[0] if candidates else df.columns[-1]


def build_pipeline(model_name: str, numeric_features, categorical_features, params: dict):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ✅ OneHotEncoder kompatibel lintas versi
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # sklearn <1.2

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
        model = GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
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

# --- DIHAPUS: Fungsi ini tidak lagi diperlukan ---
# def get_feature_names_from_ct(ct: ColumnTransformer):
#     ...
# --- AKHIR BAGIAN DIHAPUS ---

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
    test_size = 0 # Tidak dipakai jika CV aktif
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
    params["var_smoothing"] = 10 ** st.sidebar.slider("log10(var_smoothing)", -12, -6, -9)

st.sidebar.divider()
st.sidebar.caption("💾 Muat model .joblib (opsional)")
uploaded_model = st.sidebar.file_uploader("Muat Model (.joblib)", type=["joblib"], accept_multiple_files=False)

# =========================================================
# CACHE SEDERHANA: DATAFRAME
# =========================================================
df_cached = st.session_state.get("df_cached", None)

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

# -----------------------------
# Tab Data
# -----------------------------
with tab_data:
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

# -----------------------------
# Tab Train
# -----------------------------
with tab_train:
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
                            # Buat split sementara hanya untuk permutation importance
                            _, X_test_perm, _, y_test_perm = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
                        else:
                            X_test_perm, y_test_perm = X_test, y_test

                        model = pipe.named_steps["model"]
                        pre = pipe.named_steps["preprocess"]
                        
                        # --- PERBAIKAN: Gunakan metode .get_feature_names_out() ---
                        feature_names = pre.get_feature_names_out()
                        
                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                        else:
                            scorer = make_scorer(f1_score, pos_label=positive_value, zero_division=0)
                            result = permutation_importance(
                                pipe, X_test_perm, y_test_perm,
                                n_repeats=5, random_state=random_state, n_jobs=-1, scoring=scorer
                            )
                            importances = result.importances_mean

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

# =========================================================
# Chatbot Utilities
# =========================================================
REQUIRED_FEATURES = ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai", "mandiri/flagsip", "BEKERJA/TIDAK"]

FEATURE_PATTERNS = {
    "USIAMASUK": re.compile(r"(usia(\s*masuk)?|usiamasuk|umur)\s*[:=]?\s*(\d{1,2})", re.I),
    "IP2": re.compile(r"\bip(?:k|s)?\s*2\b\s*[:=]?\s*([0-4](?:[.,]\d{1,2})?)", re.I),
    "IP3": re.compile(r"\bip(?:k|s)?\s*3\b\s*[:=]?\s*([0-4](?:[.,]\d{1,2})?)", re.I),
    "IP5": re.compile(r"\bip(?:k|s)?\s*5\b\s*[:=]?\s*([0-4](?:[.,]\d{1,2})?)", re.I),
    "rata-rata nilai": re.compile(r"(rata[- ]?rata\s*nilai|nilai\s*rata[- ]?rata|rerata)\s*[:=]?\s*(\d{1,3})", re.I),
}

def extract_features_from_text(text: str, current: dict) -> dict:
    t = text.strip()
    out = dict(current)

    for key, pat in FEATURE_PATTERNS.items():
        m = pat.search(t)
        if m:
            val = m.group(m.lastindex) if m.lastindex else m.group(1)
            val = val.replace(',', '.')
            if key in {"rata-rata nilai", "USIAMASUK"}:
                try:
                    out[key] = int(float(val))
                except Exception: pass
            else:
                try:
                    out[key] = float(val)
                except Exception: pass

    for m in re.finditer(r"\bip(?:k|s)?[\s_\-]*([235])\b\s*(?:[:=：＝]|\s)\s*([0-4](?:[.,]\d{1,2})?)", t, re.I):
        idx = m.group(1)
        raw = m.group(2)
        val = raw.replace(" ", "").replace(",", ".")
        try:
            out[f"IP{idx}"] = float(val)
        except Exception: pass

    if re.search(r"\bmandiri\b", t, re.I): out["mandiri/flagsip"] = "MANDIRI"
    if re.search(r"\bflag(ship|sip)\b", t, re.I): out["mandiri/flagsip"] = "FLAGSIP"

    m_status = re.search(r"\bstatus\s*(kerja|bekerja)\s*[:=]?\s*(ya|y|true|1|tidak|tdk|t|false|0)\b", t, re.I)
    if m_status:
        v = m_status.group(2).lower()
        out["BEKERJA/TIDAK"] = "YA" if v in {"ya", "y", "true", "1"} else "TIDAK"

    assign_pairs = re.findall(r"([a-zA-Z/\- ]+)\s*=\s*([\w\.,]+)", t)
    for k_raw, v_raw in assign_pairs:
        k = _norm(k_raw)
        v = v_raw.replace(',', '.')
        if k in NAME_MAP:
            canon = NAME_MAP[k]
            if canon in {"mandiri/flagsip", "BEKERJA/TIDAK"}:
                vv = _norm(v)
                if canon == "mandiri/flagsip":
                    if vv == "mandiri": out[canon] = "MANDIRI"
                    elif vv in {"flagsip", "flagship"}: out[canon] = "FLAGSIP"
                else:
                    if vv in {"ya", "y", "true", "1"}: out[canon] = "YA"
                    elif vv in {"tidak", "tdk", "t", "false", "0"}: out[canon] = "TIDAK"
            else:
                try:
                    out[canon] = float(v) if canon not in {"USIAMASUK", "rata-rata nilai"} else int(float(v))
                except Exception: pass

    if out.get("BEKERJA/TIDAK") is None:
        if re.search(r"\b(tidak\s*(bekerja|kerja)|nggak\s*kerja|gak\s*kerja)\b", t, re.I):
            out["BEKERJA/TIDAK"] = "TIDAK"
        elif re.search(r"\b(bekerja|kerja)\b", t, re.I):
            out["BEKERJA/TIDAK"] = "YA"
    return out


def missing_features(feat: dict):
    return [f for f in REQUIRED_FEATURES if feat.get(f) in (None, "", np.nan)]


def predict_and_recommend(pipe, features_dict: dict, positive_value: str):
    one = pd.DataFrame([{k: features_dict.get(k, np.nan) for k in REQUIRED_FEATURES}])
    one = harmonize_columns(one)
    pred = pipe.predict(one)[0]
    proba_str = ""
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(one)
        classes = pipe.named_steps["model"].classes_
        pos_index = list(classes).index(positive_value) if positive_value in classes else (1 if proba.shape[1] > 1 else 0)
        p_pos = float(proba[:, pos_index][0])
        proba_str = f" (Prob positif={positive_value}: {p_pos:.3f})"

    status_bekerja = str(features_dict.get("BEKERJA/TIDAK", "")).upper()
    if str(pred).upper() == str(positive_value).upper():
        header = f"Hasil prediksi: **{pred}**{proba_str}.\n\n🎉 *Selamat! Anda diprediksi lulus tepat waktu.*"
        rekomendasi = ["- Pertahankan/tambah IP tiap semester", "- Jaga nilai rata-rata tetap tinggi"]
        if status_bekerja == "YA":
            rekomendasi.append("- Tetap fokus meski sambil bekerja")
        else:
            rekomendasi.append("- Manfaatkan waktu luang untuk kegiatan positif/akademik")
        rekomendasi.extend(["- Pilih jalur yang sesuai kemampuan", "- Konsultasi rutin dengan dosen pembimbing"])
        msg = header + "\n" + "\n".join(rekomendasi)
    else:
        header = f"Hasil prediksi: **{pred}**{proba_str}.\n\n⚠️ *Saat ini peluang lulus tepat waktu belum optimal.*"
        rekomendasi = ["- Tingkatkan IP (IP2, IP3, IP5) berikutnya", "- Upayakan nilai rata-rata naik"]
        if status_bekerja == "YA":
            rekomendasi.append("- Pertimbangkan mengurangi aktivitas luar studi jika mengganggu akademik")
        else:
             rekomendasi.append("- Fokuskan energi pada kegiatan akademik untuk hasil maksimal")
        rekomendasi.extend(["- Konsultasikan strategi belajar dengan dosen", "- Pastikan jalur (MANDIRI/FLAGSIP) sesuai"])
        msg = header + "\n" + "\n".join(rekomendasi)
    return msg


def init_chat_state():
    if "chat_messages" not in st.session_state: st.session_state["chat_messages"] = []
    if "chat_features" not in st.session_state: st.session_state["chat_features"] = {k: None for k in REQUIRED_FEATURES}


def chat_system_prompt():
    return (
        "Saya adalah asisten akademik. Berbicaralah santai. "
        "Saya dapat membantu prediksi kelulusan tepat waktu menggunakan 7 fitur: "
        "USIAMASUK, IP2, IP3, IP5, rata-rata nilai, mandiri/flagsip, BEKERJA/TIDAK. "
        "Ketik data Anda, misalnya: `usia=19 ip2=3.2 ip3=3.1 ip5=3.4 rerata=82 jalur=mandiri bekerja=tidak`. "
        "Saya akan menanyakan yang belum lengkap. Ketik `reset` untuk mulai ulang."
    )

# -----------------------------
# Tab Chatbot
# -----------------------------
with tab_chat:
    st.subheader("4) Chatbot Akademik — Tanya Jawab & Rekomendasi")
    active_model_obj = st.session_state.get("last_trained_model", None)
    if uploaded_model is not None:
        try:
            active_model_obj = joblib.load(uploaded_model)
            st.success("Model dari file berhasil dimuat (aktif untuk Chatbot).")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

    if active_model_obj is None:
        st.warning("Belum ada model aktif. Latih model di tab **Pelatihan & Evaluasi** atau muat .joblib dari sidebar.")
    else:
        init_chat_state()
        pipe = active_model_obj["pipeline"]
        positive_value = active_model_obj["positive"]

        st.markdown("**Panduan Singkat**: " + chat_system_prompt())
        for msg in st.session_state["chat_messages"]:
            try:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            except Exception:
                role = msg.get("role", "assistant").capitalize()
                st.markdown(f"**{role}:** {msg['content']}")

        user_text = None
        try:
            user_text = st.chat_input("Tulis pertanyaan atau masukkan data Anda...")
        except Exception:
            user_text = st.text_input("Tulis pertanyaan atau masukkan data Anda...", key="chat_input_fallback")
            if st.button("Kirim", key="send_btn"):
                user_text = st.session_state.get("chat_input_fallback")

        if user_text:
            st.session_state["chat_messages"].append({"role": "user", "content": user_text})
            user_low = user_text.strip().lower()
            if user_low in {"reset", "/reset", "ulang", "mulai ulang"}:
                st.session_state["chat_features"] = {k: None for k in REQUIRED_FEATURES}
                bot_reply = "State direset. Silakan kirim 7 fitur Anda atau jawab pertanyaan saya."
                st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
                st.rerun()

            feats = extract_features_from_text(user_text, st.session_state["chat_features"])
            st.session_state["chat_features"] = feats
            miss = missing_features(feats)
            if miss:
                ask_parts = []
                for m in miss:
                    if m == "USIAMASUK": ask_parts.append("USIAMASUK (angka, tahun)")
                    elif m in {"IP2", "IP3", "IP5"}: ask_parts.append(f"{m} (0.00 - 4.00)")
                    elif m == "rata-rata nilai": ask_parts.append("rata-rata nilai (0-100)")
                    elif m == "mandiri/flagsip": ask_parts.append("jalur: MANDIRI / FLAGSIP")
                    elif m == "BEKERJA/TIDAK": ask_parts.append("status kerja: YA / TIDAK")
                bot_reply = ("Data belum lengkap. Mohon lengkapi: " + ", ".join(ask_parts) + "\n\n"
                             "Contoh cepat: `usia=19 ip2=3.2 ip3=3.1 ip5=3.4 rerata=82 jalur=mandiri bekerja=tidak`")
            else:
                try:
                    bot_reply = predict_and_recommend(pipe, feats, positive_value)
                except Exception as e:
                    bot_reply = f"Maaf, terjadi kesalahan saat memprediksi: {e}"
            st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
            st.rerun()

        with st.expander("Status Fitur yang Terdeteksi"):
            st.write(st.session_state["chat_features"])
            if missing_features(st.session_state["chat_features"]):
                st.info("Lengkapi fitur yang masih kosong melalui chat.")
            else:
                st.success("Semua fitur terpenuhi. Anda bisa ketik pertanyaan lain atau `reset`.")

# -----------------------------
# Tab Form Input (7 Fitur)
# -----------------------------
with tab_form:
    st.subheader("3) Prediksi Individu — Form 7 Fitur")
    active_model_obj = st.session_state.get("last_trained_model", None)
    if uploaded_model is not None:
        try:
            active_model_obj = joblib.load(uploaded_model)
            st.success("Model dari file berhasil dimuat (aktif untuk Form).")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

    if active_model_obj is None:
        st.warning("Belum ada model aktif. Latih model di tab **Pelatihan & Evaluasi** atau muat .joblib dari sidebar.")
    else:
        pipe = active_model_obj["pipeline"]
        expected_features = active_model_obj["features"]
        positive_value = active_model_obj["positive"]
        opsi_jalur = ["MANDIRI", "FLAGSIP"]
        opsi_yn = ["YA", "TIDAK"]

        with st.form("form7"):
            colA, colB, colC = st.columns(3)
            with colA:
                USIAMASUK = st.number_input("USIAMASUK (tahun)", min_value=15, max_value=60, value=19, step=1)
                IP2 = st.number_input("IP2", min_value=0.0, max_value=4.0, value=3.2, step=0.01, format="%.2f")
            with colB:
                IP3 = st.number_input("IP3", min_value=0.0, max_value=4.0, value=3.2, step=0.01, format="%.2f")
                IP5 = st.number_input("IP5", min_value=0.0, max_value=4.0, value=3.2, step=0.01, format="%.2f")
            with colC:
                rata_rata = st.slider("rata-rata nilai", min_value=0, max_value=100, value=82, step=1)
                jalur = st.selectbox("mandiri/flagsip", opsi_jalur)
                bekerja = st.selectbox("BEKERJA/TIDAK", opsi_yn)
            submit = st.form_submit_button("🔮 Prediksi")

        if submit:
            inputs_form = {"USIAMASUK": USIAMASUK, "IP2": IP2, "IP3": IP3, "IP5": IP5, "rata-rata nilai": rata_rata, "mandiri/flagsip": jalur, "BEKERJA/TIDAK": bekerja}
            X_one = {c: (inputs_form[c] if c in inputs_form else np.nan) for c in expected_features}
            X_one = pd.DataFrame([X_one])
            try:
                pred = pipe.predict(X_one)[0]
                proba_str = ""
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    proba = pipe.predict_proba(X_one)
                    classes = pipe.named_steps["model"].classes_
                    pos_index = list(classes).index(positive_value) if positive_value in classes else (1 if proba.shape[1] > 1 else 0)
                    p_pos = float(proba[:, pos_index][0])
                    proba_str = f" — Prob(positif={positive_value}): **{p_pos:.3f}**"
                st.success(f"**Hasil Prediksi (Form 7 Fitur)**: **{pred}**{proba_str}")

                if str(pred).upper() == str(positive_value).upper():
                    tips = ["- Pertahankan atau tingkatkan IP (Indeks Prestasi) tiap semester", "- Jaga nilai rata-rata tetap tinggi"]
                    if bekerja == "YA": tips.append("- Tetap fokus pada studi walaupun sambil bekerja")
                    else: tips.append("- Manfaatkan waktu luang untuk kegiatan yang menunjang akademik")
                    tips.extend(["- Pilih jalur pendidikan yang sesuai kemampuan", "- Konsultasi rutin dengan dosen pembimbing"])
                    st.info("🎉 *Selamat! Prediksi Anda akan lulus tepat waktu.*\n\n"
                            "Tetap pertahankan kinerja Anda. Tips agar tetap di jalur:\n" + "\n".join(tips))
                else:
                    saran = ["- Tingkatkan IP di semester berikutnya (IP2, IP3, IP5)", "- Usahakan rata-rata nilai naik di semester berikutnya"]
                    if bekerja == "YA": saran.append("- Pertimbangkan mengurangi aktivitas luar studi jika mengganggu akademik")
                    else: saran.append("- Alokasikan lebih banyak waktu untuk fokus pada kegiatan akademik")
                    saran.extend(["- Konsultasikan strategi belajar dengan dosen pembimbing", "- Pastikan memilih jalur pendidikan yang sesuai", "Periksa kembali data input untuk memastikan akurasi."])
                    st.warning("⚠️ *Prediksi: Anda belum lulus tepat waktu.*\n\n"
                               "Beberapa hal yang dapat Anda tingkatkan agar peluang lulus tepat waktu lebih besar:\n" + "\n".join(saran))
                
                try:
                    if hasattr(pipe.named_steps["model"], "feature_importances_"):
                        importances = pipe.named_steps["model"].feature_importances_
                        # --- PERBAIKAN: Gunakan .get_feature_names_out() juga di sini ---
                        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
                        sorted_idx = np.argsort(importances)[::-1]
                        st.markdown("*Fitur paling berpengaruh (global):*")
                        st.write({feature_names[i]: float(importances[i]) for i in sorted_idx[:3]})
                except Exception: pass
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")

# -----------------------------
# Tab About
# -----------------------------
with tab_about:
    st.subheader("Tentang Aplikasi (Skema Terkunci)")
    st.markdown(f"""
- **Fitur digunakan**: {', '.join(CANON_FEATURES)}
- **Target**: {TARGET_NAME}
- **Catatan**:
  - Aplikasi otomatis menyamakan nama kolom dari variasi umum ke format di atas.
  - Jika ada fitur yang tidak tersedia di dataset, pelatihan tetap bisa dilakukan dengan fitur yang ada.
  - Target harus biner — Anda dapat memilih kelas **positif** di UI (mis. `TEPAT`, `YA`, atau `1`).
""")
