"""
Week 4 - AI Project Deployment & Capstone
Employee Attrition Prediction: training script

Steps
  1. load + inspect the data
  2. exploratory analysis (figures)
  3. split into train / test
  4. compare six models with repeated cross-validation
  5. tune the chosen model with GridSearchCV
  6. evaluate once on the test set (+ threshold analysis)
  7. save the fitted pipeline with joblib, plus a small model card (JSON)

Run:  python train_model.py
Output: model/attrition_model.joblib, model/model_card.json, results/, figures/
"""

import os
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, GridSearchCV,
                                     cross_validate, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)

warnings.filterwarnings("ignore")
for folder in ("model", "results", "figures"):
    os.makedirs(folder, exist_ok=True)

SEED = 42
NUM_COLS = ["Age", "Experience_Years", "Monthly_Income",
            "Satisfaction_Score", "Weekly_Work_Hours"]
CAT_COLS = ["Education", "Department", "City", "Job_Level"]
TARGET = "Attrition"


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def make_pipeline(model):
    """Preprocessing + model in one object. Whatever is learned (medians,
    means, categories) comes from the training rows only, and the same
    object is what the API loads later."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return Pipeline([("prep", prep), ("model", model)])


# ---------------------------------------------------------------
# 1. Load and inspect
# ---------------------------------------------------------------
section("1. LOAD AND INSPECT")
df = pd.read_csv("data/employee_attrition.csv")
print("shape:", df.shape, "| duplicates:", df.duplicated().sum())
print("missing values:", df.isnull().sum()[df.isnull().sum() > 0].to_dict())
df["target"] = (df[TARGET] == "Yes").astype(int)          # 1 = left
print("attrition split:", df[TARGET].value_counts().to_dict(),
      "| share leaving:", round(df["target"].mean(), 3))
print(df[NUM_COLS].describe().round(1).T[["min", "50%", "max"]].to_string())
cats = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS}
ranges = {c: [float(df[c].min()), float(df[c].max())] for c in NUM_COLS}


# ---------------------------------------------------------------
# 2. Exploratory analysis
# ---------------------------------------------------------------
section("2. EXPLORATORY ANALYSIS")
corr = df[NUM_COLS + ["target"]].corr()["target"].drop("target").round(3)
print("correlation of each numeric column with leaving:")
print(corr.sort_values().to_string())

print("\nattrition rate by satisfaction band:")
sat_bins = pd.cut(df["Satisfaction_Score"], [0, 2, 3, 4, 5], labels=["1-2", "2-3", "3-4", "4-5"])
sat_rate = df.groupby(sat_bins)["target"].mean().round(3)
print(sat_rate.to_string())
print("\nattrition rate by weekly hours band:")
hr_bins = pd.cut(df["Weekly_Work_Hours"], [0, 35, 40, 45, 50, 70],
                 labels=["<=35", "35-40", "40-45", "45-50", ">50"])
hr_rate = df.groupby(hr_bins)["target"].mean().round(3)
print(hr_rate.to_string())
print("\nattrition rate by job level:")
jl_rate = df.groupby("Job_Level")["target"].mean().round(3)
print(jl_rate.to_string())
print("\nattrition rate by city:")
city_rate = df.groupby("City")["target"].mean().round(3).sort_values()
print(city_rate.to_string())

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
ax[0, 0].bar(["stayed", "left"], df[TARGET].value_counts().reindex(["No", "Yes"]).values,
             color=["#4C72B0", "#C44E52"])
ax[0, 0].set_title("Class balance"); ax[0, 0].set_ylabel("employees")
ax[0, 1].bar(sat_rate.index.astype(str), sat_rate.values, color="#C44E52")
ax[0, 1].axhline(df["target"].mean(), color="gray", ls="--", label="overall rate")
ax[0, 1].set_title("Attrition rate by satisfaction score"); ax[0, 1].set_ylim(0, 0.9); ax[0, 1].legend()
ax[1, 0].bar(hr_rate.index.astype(str), hr_rate.values, color="#DD8452")
ax[1, 0].axhline(df["target"].mean(), color="gray", ls="--")
ax[1, 0].set_title("Attrition rate by weekly hours"); ax[1, 0].set_ylim(0, 0.9)
ax[1, 1].barh(city_rate.index, city_rate.values, color="#8172B2")
ax[1, 1].axvline(df["target"].mean(), color="gray", ls="--")
ax[1, 1].set_title("Attrition rate by city"); ax[1, 1].set_xlim(0, 0.9)
plt.tight_layout()
plt.savefig("figures/01_eda.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------
section("3. TRAIN / TEST SPLIT")
X = df[NUM_COLS + CAT_COLS]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape,
      "| share leaving: train", round(y_train.mean(), 3), "test", round(y_test.mean(), 3))


# ---------------------------------------------------------------
# 4. Compare models with repeated cross-validation (train only)
# ---------------------------------------------------------------
section("4. MODEL COMPARISON (5-fold CV x 3 repeats)")
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
candidates = {
    "Baseline (always 'left')": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                            random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                        max_iter=150, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=25),
}
rows = []
for name, mdl in candidates.items():
    r = cross_validate(make_pipeline(mdl), X_train, y_train, cv=cv_rep,
                       scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    rows.append({"Model": name,
                 "Accuracy": r["test_accuracy"].mean(),
                 "Precision": r["test_precision"].mean(),
                 "Recall": r["test_recall"].mean(),
                 "F1": r["test_f1"].mean(),
                 "ROC_AUC": r["test_roc_auc"].mean(),
                 "ROC_AUC_std": r["test_roc_auc"].std()})
cmp_df = pd.DataFrame(rows)
print(cmp_df.round(3).to_string(index=False))
cmp_df.round(4).to_csv("results/model_comparison_cv.csv", index=False)

plt.figure(figsize=(7.5, 4))
order = cmp_df.sort_values("ROC_AUC")
plt.barh(order["Model"], order["ROC_AUC"], xerr=order["ROC_AUC_std"], capsize=4,
         color=["#999999" if n.startswith("Baseline") else "#4C72B0" for n in order["Model"]])
plt.axvline(0.5, color="gray", ls=":")
plt.xlabel("cross-validated ROC-AUC (mean \u00b1 std over 15 folds)")
plt.title("Model comparison on the training data")
plt.xlim(0.4, 0.8)
plt.tight_layout()
plt.savefig("figures/02_model_comparison.png", dpi=110)
plt.close()

best_name = cmp_df.loc[cmp_df["ROC_AUC"].idxmax(), "Model"]
print("\nbest model by cross-validated ROC-AUC:", best_name)


# ---------------------------------------------------------------
# 5. Tune the selected model
# ---------------------------------------------------------------
section("5. HYPERPARAMETER TUNING (GridSearchCV)")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = {"model__C": [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100]}
gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000)), grid,
                  cv=cv5, scoring="roc_auc", n_jobs=-1, return_train_score=True)
gs.fit(X_train, y_train)
gdf = pd.DataFrame({"C": grid["model__C"],
                    "CV_ROC_AUC": gs.cv_results_["mean_test_score"],
                    "CV_std": gs.cv_results_["std_test_score"],
                    "Train_ROC_AUC": gs.cv_results_["mean_train_score"]})
print(gdf.round(4).to_string(index=False))
print("best:", gs.best_params_, "CV ROC-AUC", round(gs.best_score_, 4))
gdf.round(4).to_csv("results/hyperparameter_search.csv", index=False)

plt.figure(figsize=(6.5, 4))
plt.errorbar(range(len(gdf)), gdf["CV_ROC_AUC"], yerr=gdf["CV_std"], fmt="o-", capsize=4,
             label="cross-validation (held-out folds)")
plt.plot(range(len(gdf)), gdf["Train_ROC_AUC"], "s--", color="#C44E52", label="training folds")
plt.xticks(range(len(gdf)), gdf["C"])
plt.xlabel("C (smaller = stronger regularization)"); plt.ylabel("ROC-AUC")
plt.title("Tuning the logistic regression"); plt.legend()
plt.tight_layout()
plt.savefig("figures/06_tuning_curve.png", dpi=110)
plt.close()
model = gs.best_estimator_


# ---------------------------------------------------------------
# 6. Evaluate once on the test set
# ---------------------------------------------------------------
section("6. TEST SET EVALUATION")
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {"accuracy": accuracy_score(y_test, pred),
           "precision": precision_score(y_test, pred),
           "recall": recall_score(y_test, pred),
           "f1": f1_score(y_test, pred),
           "roc_auc": roc_auc_score(y_test, prob)}
base_acc = max(y_test.mean(), 1 - y_test.mean())
base_f1 = f1_score(y_test, np.ones_like(y_test))
print({k: round(v, 3) for k, v in metrics.items()})
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"baseline (always 'left'): accuracy {base_acc:.3f}, F1 {base_f1:.3f}")

# uncertainty: bootstrap the test set
rng = np.random.RandomState(SEED)
yt, pt = y_test.to_numpy(), prob
boot_auc, boot_acc = [], []
for _ in range(2000):
    i = rng.randint(0, len(yt), len(yt))
    if yt[i].min() != yt[i].max():
        boot_auc.append(roc_auc_score(yt[i], pt[i]))
        boot_acc.append(accuracy_score(yt[i], pt[i] >= 0.5))
auc_ci = np.percentile(boot_auc, [2.5, 97.5])
acc_ci = np.percentile(boot_acc, [2.5, 97.5])
print(f"ROC-AUC 95% interval: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
print(f"accuracy 95% interval: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")

pd.DataFrame([{**metrics, "baseline_accuracy": base_acc, "baseline_f1": base_f1,
               "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1]}]).round(4).to_csv(
    "results/test_metrics.csv", index=False)
pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual stayed", "Actual left"],
             columns=["Predicted stayed", "Predicted left"]).to_csv("results/confusion_matrix.csv")

# confusion matrix + ROC curve
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
cm = np.array([[tn, fp], [fn, tp]])
ax[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(["stayed", "left"]); ax[0].set_yticklabels(["stayed", "left"])
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual")
ax[0].set_title("Confusion matrix (test set, threshold 0.50)")
fpr, tpr, _ = roc_curve(y_test, prob)
ax[1].plot(fpr, tpr, color="#C44E52", lw=2, label=f"model (AUC {metrics['roc_auc']:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", label="random guessing")
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
ax[1].set_title("ROC curve (test set)"); ax[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/03_confusion_and_roc.png", dpi=110)
plt.close()

# threshold trade-off from out-of-fold training predictions
oof = cross_val_predict(model, X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
ths = np.linspace(0.2, 0.8, 25)
tr = pd.DataFrame({"threshold": ths,
                   "precision": [precision_score(y_train, oof >= t) for t in ths],
                   "recall": [recall_score(y_train, oof >= t) for t in ths],
                   "accuracy": [accuracy_score(y_train, oof >= t) for t in ths]})
tr.round(4).to_csv("results/threshold_tradeoff.csv", index=False)
print("\nthreshold trade-off (out-of-fold, training data):")
print(tr.iloc[::4].round(3).to_string(index=False))
plt.figure(figsize=(6.5, 4))
for col, c in [("precision", "#4C72B0"), ("recall", "#C44E52"), ("accuracy", "#55A868")]:
    plt.plot(tr.threshold, tr[col], label=col, color=c, lw=2)
plt.axvline(0.5, color="gray", ls=":")
plt.xlabel("decision threshold"); plt.title("Threshold trade-off (out-of-fold)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/05_threshold_tradeoff.png", dpi=110)
plt.close()

# what the model learned
names = [n.split("__")[1] for n in model.named_steps["prep"].get_feature_names_out()]
coefs = pd.Series(model.named_steps["model"].coef_[0], index=names).sort_values()
print("\nlargest coefficients (positive = more likely to leave):")
print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(3).to_string())
coefs.round(4).to_csv("results/coefficients.csv", header=["coefficient"])
top = pd.concat([coefs.head(6), coefs.tail(6)]).drop_duplicates()
plt.figure(figsize=(7, 4.8))
plt.barh(top.index, top.values, color=["#4C72B0" if v < 0 else "#C44E52" for v in top.values])
plt.axvline(0, color="black", lw=0.8)
plt.xlabel("coefficient (red = raises attrition risk, blue = lowers it)")
plt.title("Logistic regression: strongest drivers")
plt.tight_layout()
plt.savefig("figures/04_coefficients.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 7. Save the model and a model card
# ---------------------------------------------------------------
section("7. SAVE MODEL (joblib)")
model.fit(X_train, y_train)                    # refit is identical; keeps things explicit
joblib.dump(model, "model/attrition_model.joblib")
card = {
    "model": "Logistic Regression in a scikit-learn Pipeline (impute -> scale / one-hot -> model)",
    "trained_on": str(date.today()),
    "sklearn_version": sklearn.__version__,
    "data": {"rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
             "synthetic": True},
    "features": {"numeric": NUM_COLS, "categorical": CAT_COLS},
    "allowed_categories": cats,
    "numeric_ranges_seen": ranges,
    "best_params": {k: float(v) for k, v in gs.best_params_.items()},
    "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    "test_roc_auc_95_interval": [round(float(auc_ci[0]), 3), round(float(auc_ci[1]), 3)],
    "baseline_accuracy": round(float(base_acc), 4),
    "intended_use": "Educational demonstration of an ML deployment. Not for employment decisions.",
}
with open("model/model_card.json", "w") as f:
    json.dump(card, f, indent=2)
print("saved model/attrition_model.joblib and model/model_card.json")
print("file size (KB):", round(os.path.getsize("model/attrition_model.joblib") / 1024, 1))

# quick reload check, like the API will do
reloaded = joblib.load("model/attrition_model.joblib")
same = np.allclose(reloaded.predict_proba(X_test)[:, 1], prob)
print("reloaded model gives identical predictions:", same)

section("DONE")
