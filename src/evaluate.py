"""
Benchmark: classify resumes into 25 job categories with TF-IDF + linear models,
and show how duplicate resumes inflate the accuracy most tutorials report.
Data: UpdatedResumeDataSet.csv (962 rows, 25 categories; Kaggle "Resume Dataset").
Usage: python src/evaluate.py UpdatedResumeDataSet.csv reports
"""
import json, re, sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SRC)
clean = lambda t: re.sub(r"\s+", " ", re.sub(r"http\S+|[^A-Za-z0-9+#. ]", " ", str(t))).lower().strip()
df["text"] = df["Resume"].map(clean)
pipe = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
                             LinearSVC(C=0.5))

# 1) The common (leaky) setup: random split over all 962 rows, many of them duplicates.
Xtr, Xte, ytr, yte = train_test_split(df.text, df.Category, test_size=0.2, random_state=7, stratify=df.Category)
leaky = pipe().fit(Xtr, ytr).predict(Xte)
dupes_in_test = float(Xte.isin(set(Xtr)).mean())

# 2) The honest setup: deduplicate, then 3-fold stratified cross-validation (smallest class has 3 resumes).
dd = df.drop_duplicates("text").reset_index(drop=True)
pred = cross_val_predict(pipe(), dd.text, dd.Category, cv=StratifiedKFold(3, shuffle=True, random_state=7))
res = {"rows": len(df), "unique_resumes": len(dd), "categories": int(df.Category.nunique()),
       "leaky_random_split": {"accuracy": round(accuracy_score(yte, leaky), 3),
                              "macro_f1": round(f1_score(yte, leaky, average="macro"), 3),
                              "share_of_test_resumes_also_in_train": round(dupes_in_test, 3)},
       "deduplicated_cv": {"accuracy": round(accuracy_score(dd.Category, pred), 3),
                           "macro_f1": round(f1_score(dd.Category, pred, average="macro"), 3),
                           "folds": 3}}
json.dump(res, open(OUT / "benchmark.json", "w"), indent=1)
print(json.dumps(res, indent=1))
