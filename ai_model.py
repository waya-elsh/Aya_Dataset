import pandas as pd
import librosa
import numpy as np
from collections import Counter
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

CSV_PATH = "MetadataAya.csv"
MAX_FRAMES = 40
RANDOM_STATE = 42
RATIO_CORRECT_TO_INCORRECT = 3

np.random.seed(RANDOM_STATE)

# ==========================
# LOAD METADATA
# ==========================
metadata = pd.read_csv(CSV_PATH, sep=';')
metadata.columns = metadata.iloc[0]
metadata = metadata[1:].reset_index(drop=True)
metadata.columns = metadata.columns.str.strip()

X_list = []
y_list = []
word_list = []

# ==========================
# FEATURE EXTRACTION
# ==========================
for _, row in metadata.iterrows():
    file_path = row["File path"]
    label = row["Label"]
    word_id = row["Word id"]

    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        mfcc = librosa.feature.mfcc(
            y=audio_trimmed,
            sr=sr,
            n_mfcc=13
        )

        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_FRAMES]

        X_list.append(mfcc)
        y_list.append(label)
        word_list.append(str(word_id).strip())

    except Exception as e:
        print("Skipped:", file_path)
        print(e)

X = np.array(X_list)
y = np.array([label.strip().lower() for label in y_list])
word_array = np.array(word_list)

label_map = {
    "correct": 1,
    "incorrect": 0
}
y_encoded = np.array([label_map[label] for label in y])

# stratified by word + label
stratify_col = np.array([
    f"{word}_{label}"
    for word, label in zip(word_array, y)
])

print("Minimum group count:", min(Counter(stratify_col).values()))

# ==========================
# STRATIFIED 5-FOLD LOGISTIC
# ==========================
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []
auc_scores = []
sens_scores = []
spec_scores = []
all_conf_matrices = []

best_f1 = 0
best_model = None
best_scaler = None

roc_y_true = []
roc_y_prob = []

fold_num = 1

for train_idx, test_idx in skf.split(X, stratify_col):

    print(f"\n==============================")
    print(f"FOLD {fold_num}")
    print("==============================")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print("Train size:", len(y_train))
    print("Test size:", len(y_test))
    print("Before balancing:", Counter(y_train))

    # -------- downsampling --------
    correct_idx = np.where(y_train == 1)[0]
    incorrect_idx = np.where(y_train == 0)[0]

    incorrect_count = len(incorrect_idx)
    target_correct_count = incorrect_count * RATIO_CORRECT_TO_INCORRECT

    if len(correct_idx) > target_correct_count:
        correct_sampled = np.random.choice(correct_idx, target_correct_count, replace=False)
    else:
        correct_sampled = correct_idx

    balanced_idx = np.concatenate([correct_sampled, incorrect_idx])
    np.random.shuffle(balanced_idx)

    X_train_bal = X_train[balanced_idx]
    y_train_bal = y_train[balanced_idx]

    print("After balancing:", Counter(y_train_bal))

    # -------- flatten --------
    X_train_flat = X_train_bal.reshape(X_train_bal.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # -------- scaling --------
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    # -------- class weight --------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_bal),
        y=y_train_bal
    )
    class_weight_dict = dict(zip(np.unique(y_train_bal), class_weights))

    # -------- logistic model --------
    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight=class_weight_dict,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_scaled, y_train_bal)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # -------- metrics --------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy    :", round(acc, 4))
    print("Precision   :", round(prec, 4))
    print("Recall      :", round(rec, 4))
    print("F1 Score    :", round(f1, 4))
    print("Sensitivity :", round(sensitivity, 4))
    print("Specificity :", round(specificity, 4))
    print("ROC AUC     :", round(auc, 4))

    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)
    auc_scores.append(auc)
    sens_scores.append(sensitivity)
    spec_scores.append(specificity)
    all_conf_matrices.append(cm)

    roc_y_true.extend(y_test)
    roc_y_prob.extend(y_prob)

    # -------- save best fold --------
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_scaler = scaler

    fold_num += 1

# ==========================
# FINAL RESULTS
# ==========================
print("\n==============================")
print("FINAL LOGISTIC RESULTS")
print("==============================")

print("Mean Accuracy    :", round(np.mean(acc_scores), 4))
print("Mean Precision   :", round(np.mean(prec_scores), 4))
print("Mean Recall      :", round(np.mean(rec_scores), 4))
print("Mean F1 Score    :", round(np.mean(f1_scores), 4))
print("Mean Sensitivity :", round(np.mean(sens_scores), 4))
print("Mean Specificity :", round(np.mean(spec_scores), 4))
print("Mean ROC AUC     :", round(np.mean(auc_scores), 4))

total_cm = np.sum(all_conf_matrices, axis=0)
print("\nSummed Confusion Matrix:")
print(total_cm)

# ==========================
# ROC CURVE PLOT
# ==========================
fpr, tpr, thresholds = roc_curve(roc_y_true, roc_y_prob)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label="Logistic Regression ROC Curve")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()

# ==========================
# SAVE BEST MODEL
# ==========================
joblib.dump(best_model, "final_model.pkl")
joblib.dump(best_scaler, "final_scaler.pkl")

# ==========================================
# SAVE TRAINING MFCC REFERENCE CENTER
# ==========================================
X_all_flat = X.reshape(X.shape[0], -1)
X_all_scaled = best_scaler.transform(X_all_flat)

mfcc_reference = np.mean(X_all_scaled, axis=0)
np.save("mfcc_reference.npy", mfcc_reference)

print("Best fold model saved successfully.")
print("Saved files:")
print("final_model.pkl")
print("final_scaler.pkl")
print("mfcc_reference.npy")
print("roc_curve.png")