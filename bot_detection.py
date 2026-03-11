# ============================
# ONE-CELL COLAB — Robust Kaggle + Fallbacks
# Hybrid ML–NLP Stacked Ensemble for Twitter Bot Detection
# Primary target: charlesburell/twitter-bots-dataset
# Fallbacks: mtesconi/twitter-bot-detection, goyaladi/twitter-bot-detection-dataset,
#            davidmartngutirrez/twitter-bots-accounts, juice0lover/users-vs-bots-classification
# If Kaggle API is forbidden (403), it will prompt you to upload the ZIP/CSV manually.
# ============================

# 0) Installs
!pip -q install --upgrade pip
!pip -q install pandas numpy scikit-learn xgboost==2.0.3 imbalanced-learn matplotlib textblob nltk kaggle

import os, glob, re, warnings, json, sys, shutil, subprocess, textwrap
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import nltk
nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')

from google.colab import files
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

def shell(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

# 1) Kaggle auth (robust: accepts any uploaded filename; installs to ~/.kaggle/kaggle.json)
print("➡️ Upload your Kaggle API token (Account → Create New Token).")
uploaded = files.upload()
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
if uploaded:
    up_name = list(uploaded.keys())[0]
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
        f.write(uploaded[up_name])
!chmod 600 ~/.kaggle/kaggle.json

# 2) Try multiple Kaggle slugs in order
SLUGS = [
    "charlesburell/twitter-bots-dataset",         # your requested
    "mtesconi/twitter-bot-detection",             # classic
    "goyaladi/twitter-bot-detection-dataset",     # public alt
    "davidmartngutirrez/twitter-bots-accounts",   # small alt
    "juice0lover/users-vs-bots-classification"    # small alt
]

print("\n🔎 Attempting Kaggle download...")
os.makedirs("/content/data", exist_ok=True)
downloaded = False
for slug in SLUGS:
    print(f"📥 Trying: {slug}")
    r = shell(f'kaggle datasets download -d "{slug}" -p /content/data -q --force')
    if r.returncode == 0:
        # unzip if any zip found
        uz = shell('unzip -o -q /content/data/*.zip -d /content/data || true')
        print(f"✅ Downloaded: {slug}")
        downloaded = True
        break
    else:
        print(f"❌ Kaggle error for {slug}: {r.stderr.strip()[:200]}")
        # If 403, user must accept rules or dataset is private
        if "403" in r.stderr:
            print("   → 403 Forbidden (accept dataset rules or dataset is private). Trying next fallback...")

print("\n📂 Files in /content/data (after Kaggle attempts):")
shell('ls -lah /content/data | sed -n "1,200p"')
csvs = glob.glob("/content/data/*.csv") + glob.glob("/content/data/**/*.csv", recursive=True)

# 3) If no CSVs, prompt for manual upload (ZIP or CSV)
if not csvs:
    print("\n⚠️ No CSV found from Kaggle. Two quick fixes:")
    print("   A) MANUAL: Download ZIP/CSV from Kaggle web → Upload here; or")
    print("   B) Accept dataset rules in the Kaggle page while logged in, then re-run.")
    print("\nDirect dataset pages (open, click **Download** once to accept rules):")
    print(" • Twitter Bots Dataset — user/profile features + labels:")
    print("   https://www.kaggle.com/datasets/charlesburell/twitter-bots-dataset")
    print(" • Twitter-Bot Detection (honeypot-style):")
    print("   https://www.kaggle.com/datasets/mtesconi/twitter-bot-detection")
    print(" • Twitter Bot Detection Dataset:")
    print("   https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset")
    print(" • Twitter Bots Accounts (minimal):")
    print("   https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts")
    print(" • Users vs Bots Classification:")
    print("   https://www.kaggle.com/datasets/juice0lover/users-vs-bots-classification")

    print("\n⬆️ Please upload the ZIP or CSV now...")
    up2 = files.upload()
    # If zip uploaded, unzip; else if csv uploaded, just proceed
    for fname in up2:
        if fname.lower().endswith(".zip"):
            shell(f'unzip -o -q "{fname}" -d /content/data')
        elif fname.lower().endswith(".csv"):
            shutil.move(fname, "/content/data/")
    csvs = glob.glob("/content/data/*.csv") + glob.glob("/content/data/**/*.csv", recursive=True)
    assert len(csvs) > 0, "Still no CSV found. Please ensure you uploaded a .zip or .csv from the dataset."

# 4) Load CSV (pick first CSV found)
DATA_FILE = csvs[0]
print("\nUsing CSV:", DATA_FILE)
df = pd.read_csv(DATA_FILE, encoding="utf-8", engine="python")
print("Shape:", df.shape)
display(df.head(3))

# 5) Column mapping (text/label/numerics)
def pick_columns(df):
    text_candidates = [c for c in df.columns if any(k in c.lower() for k in
                        ["text","tweet","description","bio","content","status","message"])]
    label_candidates = [c for c in df.columns if any(k in c.lower() for k in
                        ["label","bot","isbot","target","class"])]
    numeric_candidates = [c for c in df.columns if df[c].dtype.kind in "if"]
    TEXT_COL = text_candidates[0] if text_candidates else df.columns[0]
    LABEL_COL = label_candidates[0] if label_candidates else df.columns[-1]
    NUMERIC_COLS = [c for c in numeric_candidates if c != LABEL_COL]
    if not NUMERIC_COLS:
        common = [c for c in df.columns if any(k in c.lower() for k in
                   ["followers","friends","favourites","listed","statuses","retweet","reply","post","tweet_count","like","favourites_count","statuses_count"])]
        NUMERIC_COLS = [c for c in common if c in df.columns]
        if not NUMERIC_COLS:
            NUMERIC_COLS = [c for c in df.select_dtypes(include=[np.number]).columns if c != LABEL_COL]
    return TEXT_COL, LABEL_COL, NUMERIC_COLS

TEXT_COL, LABEL_COL, NUMERIC_COLS = pick_columns(df)
print(f"\nMapped TEXT_COL='{TEXT_COL}', LABEL_COL='{LABEL_COL}'")
print("Numeric columns (subset):", NUMERIC_COLS[:12])

# 6) Label → binary
def to_binary(v):
    s = str(v).strip().lower()
    if s in {"1","true","bot","fake","spam","malicious","automated"}: return 1
    if s in {"0","false","human","genuine","real"}: return 0
    try: return 1 if float(s) >= 0.5 else 0
    except: return 1 if "bot" in s or "fake" in s else 0
y = df[LABEL_COL].map(to_binary).astype(int)

# 7) Split
df[TEXT_COL] = df[TEXT_COL].fillna("")
X_tab = df[NUMERIC_COLS].copy()
X_text = df[[TEXT_COL]].copy()
X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_tab, X_text, y, test_size=0.2, random_state=42, stratify=y
)
print("Train/Test sizes:", X_tab_train.shape, X_tab_test.shape, y_train.shape, y_test.shape)

# 8) NLP: Normalize + TF–IDF
stop_en = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)
lemm = WordNetLemmatizer()
def normalize_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"[@#]\w+", " TAG ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(lemm.lemmatize(w) for w in s.split() if w not in stop_en and len(w) > 2)
    return s

X_text_train_norm = X_text_train[TEXT_COL].astype(str).map(normalize_text)
X_text_test_norm  = X_text_test[TEXT_COL].astype(str).map(normalize_text)
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,2), min_df=3)
X_tfidf_train = tfidf.fit_transform(X_text_train_norm)
X_tfidf_test  = tfidf.transform(X_text_test_norm)
print("TF-IDF shapes:", X_tfidf_train.shape, X_tfidf_test.shape)

# 9) Base learners
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42,
    class_weight="balanced_subsample"
)
xgb_clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1
)
lr_text = LogisticRegression(C=2.0, penalty="l2", solver="liblinear", max_iter=200, class_weight="balanced")

rf.fit(X_tab_train.fillna(0), y_train)
xgb_clf.fit(X_tab_train.fillna(0), y_train)
lr_text.fit(X_tfidf_train, y_train)
print("✅ Base models trained")

# 10) Stacking with OOF meta-features
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
oof_rf = np.zeros(len(y_train)); oof_xgb = np.zeros(len(y_train)); oof_lr = np.zeros(len(y_train))
X_tab_train_f = X_tab_train.fillna(0).values; y_train_arr = y_train.values

for fold, (tr, vl) in enumerate(skf.split(X_tab_train_f, y_train_arr)):
    Xtr_tab, Xvl_tab = X_tab_train_f[tr], X_tab_train_f[vl]; ytr, yvl = y_train_arr[tr], y_train_arr[vl]
    rf_k = RandomForestClassifier(n_estimators=350, n_jobs=-1, random_state=fold, class_weight="balanced_subsample")
    rf_k.fit(Xtr_tab, ytr); oof_rf[vl] = rf_k.predict_proba(Xvl_tab)[:,1]
    xgb_k = xgb.XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=fold, n_jobs=-1
    )
    xgb_k.fit(Xtr_tab, ytr); oof_xgb[vl] = xgb_k.predict_proba(Xvl_tab)[:,1]
    Xtr_tfidf = X_tfidf_train[tr]; Xvl_tfidf = X_tfidf_train[vl]
    lr_k = LogisticRegression(C=2.0, penalty="l2", solver="liblinear", max_iter=200, class_weight="balanced")
    lr_k.fit(Xtr_tfidf, ytr); oof_lr[vl] = lr_k.predict_proba(Xvl_tfidf)[:,1]

Z_train = np.vstack([oof_rf, oof_xgb, oof_lr]).T
meta = LogisticRegression(C=2.0, penalty="l2", solver="liblinear", max_iter=200)
meta.fit(Z_train, y_train_arr)

# 11) Evaluate
p_rf_test  = rf.predict_proba(X_tab_test.fillna(0))[:,1]
p_xgb_test = xgb_clf.predict_proba(X_tab_test.fillna(0))[:,1]
p_lr_test  = lr_text.predict_proba(X_tfidf_test)[:,1]
Z_test = np.vstack([p_rf_test, p_xgb_test, p_lr_test]).T
p_final = meta.predict_proba(Z_test)[:,1]; y_pred  = (p_final >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
ap = average_precision_score(y_test, p_final)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== EVALUATION ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | PR-AUC: {ap:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# 12) Save artifacts
os.makedirs("/content/artifacts", exist_ok=True)
joblib.dump(tfidf, "/content/artifacts/tfidf.joblib")
joblib.dump(rf, "/content/artifacts/model_rf.joblib")
joblib.dump(xgb_clf, "/content/artifacts/model_xgb.joblib")
joblib.dump(lr_text, "/content/artifacts/model_lr_text.joblib")
joblib.dump(meta, "/content/artifacts/model_meta.joblib")
print("\n✅ Saved artifacts to /content/artifacts")

# 13) Predict helper
def normalize_text_for_pred(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"[@#]\w+", " TAG ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(WordNetLemmatizer().lemmatize(w) for w in s.split()
                 if w not in stopwords.words("english") and w not in ENGLISH_STOP_WORDS and len(w) > 2)
    return s

def predict_single(tab_row: pd.Series, text_value: str, threshold=0.5):
    tab_vec = tab_row[NUMERIC_COLS].fillna(0).values.reshape(1, -1)
    text_norm = normalize_text_for_pred(str(text_value))
    X_t = tfidf.transform([text_norm])
    prf  = rf.predict_proba(tab_vec)[:,1]
    pxgb = xgb_clf.predict_proba(tab_vec)[:,1]
    plr  = lr_text.predict_proba(X_t)[:,1]
    Z    = np.vstack([prf, pxgb, plr]).T
    p    = meta.predict_proba(Z)[:,1][0]
    return ("Fake" if p >= threshold else "Genuine"), float(p)

# Demo
try:
    #fake_positions = np.where(y_pred == 1)[0]
    #idx = fake_positions[0]
    demo_row = X_tab_test.iloc[0]; demo_text = X_text_test.iloc[0][TEXT_COL]
    label, score = predict_single(demo_row, demo_text, threshold=0.5)
    print("\nDemo prediction →", label, "| Score:", round(score, 4))
except Exception as e:
    print("Demo prediction skipped:", e)

print("\nAll done ✅")
