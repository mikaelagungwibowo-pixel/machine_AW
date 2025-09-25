# app.py
from io import BytesIO
import io
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
    f1_score, make_scorer,  # ✅ scorer F1 kustom dengan pos_label
)
from sklearn.model_selection import train_test_split
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
            if pd.isna(v): return v
            s = str(v).strip().lower()
            if s in {"1", "ya", "y", "true", "bekerja"}: return "YA"
            if s in {"0", "tidak", "tdk", "t", "false", "tidak bekerja"}: return "TIDAK"
            return str(v).upper()
        df["BEKERJA/TIDAK"] = df["BEKERJA/TIDAK"].apply(norm_bin_work)

    # 3) Pastikan numerik benar-benar numerik
    for col in ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai"]:
        if col in df.columns:
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
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn <1.2

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

    if model_name == "Naive Bayes (GaussianNB)":
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

def get_feature_names_from_ct(ct: ColumnTransformer):
    output_features = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    fn = last_step.get_feature_names_out(cols)
                except Exception:
                    fn = last_step.get_feature_names_out()
                output_features.extend(fn)
            else:
                output_features.extend(cols if isinstance(cols, list) else [cols])
        else:
            output_features.extend(cols if isinstance(cols, list) else [cols])
    return output_features

def to_binary(y_series: pd.Series, positive_value):
    return (y_series == positive_value).astype(int)

# =========================================================
# SIDEBAR PARAMETER
# =========================================================
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.caption("Model & parameter pelatihan")

model_name = st.sidebar.selectbox(
    "Pilih Model", ["Random Forest", "Decision Tree", "Naive Bayes (GaussianNB)"], index=0
)
test_size = st.sidebar.slider("Porsi Test Set", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

params = {"random_state": random_state}
if model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 100, 1000, 300, 50)
    params["max_depth"] = st.sidebar.select_slider("max_depth", options=[None, 5, 10, 15, 20, 30, 50], value=None)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    params["balanced"] = st.sidebar.toggle("class_weight='balanced'", value=True)
elif model_name == "Decision Tree":
    params["max_depth"] = st.sidebar.select_slider("max_depth", options=[None, 3, 5, 10, 15, 20, 30], value=None)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    params["balanced"] = st.sidebar.toggle("class_weight='balanced'", value=True)
elif model_name == "Naive Bayes (GaussianNB)":
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

tab_data, tab_train, tab_form, tab_about = st.tabs(
    ["📁 Data", "🏋️ Pelatihan & Evaluasi", "📝 Form Input (7 Fitur)", "ℹ️ Tentang"]
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
    # Template Excel
    st.markdown("**⬇️ Unduh Template Excel (header sesuai skema)**")
    sample_rows = [
        {"USIAMASUK": 18, "IP2": 3.2, "IP3": 3.3, "IP5": 3.4, "rata-rata nilai": 82, "mandiri/flagsip": "MANDIRI", "BEKERJA/TIDAK": "TIDAK", "LULUS TEPAT/TIDAK": "TEPAT"},
        {"USIAMASUK": 19, "IP2": 3.0, "IP3": 3.0, "IP5": 2.9, "rata-rata nilai": 75, "mandiri/flagsip": "FLAGSIP", "BEKERJA/TIDAK": "YA", "LULUS TEPAT/TIDAK": "TIDAK"},
    ]
    tpl_df = pd.DataFrame(sample_rows)
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        tpl_df.to_excel(writer, index=False, sheet_name="DATA")
    st.download_button(
        "Unduh Template Excel (sample_data_skematerkunci.xlsx)",
        data=excel_buf.getvalue(),
        file_name="sample_data_skematerkunci.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                default_positive = ("TEPAT" if "TEPAT" in unique_target_vals
                                    else ("YA" if "YA" in unique_target_vals
                                          else (1 if 1 in unique_target_vals else unique_target_vals[0])))
                positive_value = st.selectbox(
                    "Nilai target yang dianggap **positif** (Kelulusan Tepat Waktu)",
                    options=unique_target_vals,
                    index=unique_target_vals.index(default_positive) if default_positive in unique_target_vals else 0
                )

                run_train = st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True)
                if run_train:
                    X = df[locked_features].copy()
                    y = df[target_col].copy()
                    y_bin = to_binary(y, positive_value)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state,
                        stratify=y_bin if len(np.unique(y_bin)) == 2 else None
                    )
                    y_test_bin = to_binary(y_test, positive_value)

                    pipe = build_pipeline(model_name, numeric_features, categorical_features, params)
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                    # Probabilitas (jika tersedia)
                    y_score = None
                    if hasattr(pipe.named_steps["model"], "predict_proba"):
                        try:
                            proba = pipe.predict_proba(X_test)
                            classes = pipe.named_steps["model"].classes_
                            pos_index = list(classes).index(positive_value) if positive_value in classes else (1 if proba.shape[1] > 1 else 0)
                            y_score = proba[:, pos_index]
                        except Exception:
                            y_score = None

                    acc = accuracy_score(y_test, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary", pos_label=positive_value, zero_division=0
                    )

                    st.success(f"**Evaluasi (Test Set)** — Accuracy: **{acc:.3f}** | Precision: **{prec:.3f}** | Recall: **{rec:.3f}** | F1: **{f1:.3f}**")
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, y_pred, zero_division=0))

                    labels_for_cm = list(dict.fromkeys([positive_value] + [v for v in unique_target_vals if v != positive_value]))
                    cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm[:2] if len(labels_for_cm) >= 2 else labels_for_cm)
                    st.markdown("**Confusion Matrix**")
                    try:
                        plot_confusion_matrix(cm, labels=[str(positive_value), "NEGATIF"])
                    except Exception:
                        st.write(cm)

                    if y_score is not None and len(np.unique(y_test_bin)) == 2:
                        st.markdown("**ROC & PR Curve**")
                        plot_roc_pr(y_test_bin, y_score)
                    else:
                        st.info("ROC/PR tidak tersedia (model tidak menyediakan probabilitas atau target tidak biner).")

                    # Pentingnya Fitur
                    st.markdown("**Pentingnya Fitur**")
                    try:
                        model = pipe.named_steps["model"]
                        pre = pipe.named_steps["preprocess"]
                        feature_names = get_feature_names_from_ct(pre)

                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                        else:
                            # ✅ Gunakan F1 scorer dengan pos_label=positive_value
                            scorer = make_scorer(f1_score, pos_label=positive_value, zero_division=0)
                            result = permutation_importance(
                                pipe, X_test, y_test,
                                n_repeats=5,
                                random_state=random_state,
                                n_jobs=-1,
                                scoring=scorer
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

                    # Simpan model
                    st.markdown("**💾 Simpan Model**")
                    buf = io.BytesIO()
                    joblib.dump({"pipeline": pipe, "features": locked_features, "target": target_col, "positive": positive_value}, buf)
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
            inputs_form = {
                "USIAMASUK": USIAMASUK,
                "IP2": IP2,
                "IP3": IP3,
                "IP5": IP5,
                "rata-rata nilai": rata_rata,
                "mandiri/flagsip": jalur,
                "BEKERJA/TIDAK": bekerja,
            }

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
                    st.info(
                "🎉 *Selamat! Prediksi Anda akan lulus tepat waktu.*\n\n"
                "Tetap pertahankan kinerja Anda. Tips agar tetap di jalur:\n"
                "- Pertahankan atau tingkatkan IP (Indeks Prestasi) tiap semester\n"
                "- Jaga nilai rata-rata tetap tinggi\n"
                "- Tetap fokus pada studi walaupun sambil bekerja\n"
                "- Pilih jalur pendidikan yang sesuai kemampuan\n"
                "- Konsultasi rutin dengan dosen pembimbing"
            )
        else:
            st.warning(
                "⚠️ *Prediksi: Anda belum lulus tepat waktu.*\n\n"
                "Beberapa hal yang dapat Anda tingkatkan agar peluang lulus tepat waktu lebih besar:\n"
                "- Tingkatkan IP di semester berikutnya (IP2, IP3, IP5)\n"
                "- Usahakan rata-rata nilai naik di semester berikutnya\n"
                "- Pertimbangkan mengurangi aktivitas luar studi jika mengganggu akademik\n"
                "- Konsultasikan strategi belajar dengan dosen pembimbing\n"
                "- Pastikan memilih jalur pendidikan yang sesuai\n"
                "Periksa kembali data input untuk memastikan akurasi."
            )
        # ==== OPTIONAL: tampilkan fitur paling berpengaruh jika model mendukung ====
        try:
            if hasattr(pipe.named_steps["model"], "feature_importances_"):
                importances = pipe.named_steps["model"].feature_importances_
                feature_names = get_feature_names_from_ct(pipe.named_steps["preprocess"])
                sorted_idx = np.argsort(importances)[::-1]
                st.markdown("*Fitur paling berpengaruh (global):*")
                st.write({feature_names[i]: float(importances[i]) for i in sorted_idx[:3]})
        except Exception:
            pass
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



