import pandas as pd
import librosa
import numpy as np
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

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
y = np.array(y_list)
word_array = np.array(word_list)

y = np.array([label.strip().lower() for label in y])

label_map = {
    "correct": 1,
    "incorrect": 0
}
y_encoded = np.array([label_map[label] for label in y])

stratify_col = np.array([
    f"{word}_{label}"
    for word, label in zip(word_array, y)
])

group_counts = Counter(stratify_col)
print("Minimum group count:", min(group_counts.values()))

# ==========================
# MODELS
# ==========================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE
    )
}

# ==========================
# STRATIFIED 5 FOLD
# ==========================
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

final_results = []

for model_name, base_model in models.items():
    model_conf_matrices = []

    print(f"\n==============================")
    print(f"MODEL: {model_name}")
    print(f"==============================")

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    fold_num = 1

    for train_idx, test_idx in skf.split(X, stratify_col):

        print(f"\n----- Fold {fold_num} -----")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        print("Train size:", len(y_train))
        print("Test size:", len(y_test))
        print("Before balancing:", Counter(y_train))

        correct_idx = np.where(y_train == 1)[0]
        incorrect_idx = np.where(y_train == 0)[0]

        incorrect_count = len(incorrect_idx)
        target_correct_count = incorrect_count * RATIO_CORRECT_TO_INCORRECT

        if len(correct_idx) > target_correct_count:
            correct_sampled = np.random.choice(
                correct_idx,
                target_correct_count,
                replace=False
            )
        else:
            correct_sampled = correct_idx

        incorrect_sampled = incorrect_idx

        balanced_idx = np.concatenate([correct_sampled, incorrect_sampled])
        np.random.shuffle(balanced_idx)

        X_train_bal = X_train[balanced_idx]
        y_train_bal = y_train[balanced_idx]

        print("After ratio sampling:", Counter(y_train_bal))

        X_train_flat = X_train_bal.reshape(X_train_bal.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        X_train_scaled = np.nan_to_num(X_train_scaled)
        X_test_scaled = np.nan_to_num(X_test_scaled)

        classes = np.unique(y_train_bal)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train_bal
        )
        class_weight_dict = dict(zip(classes, class_weights))

        if model_name in ["Logistic Regression", "SVM", "Random Forest", "Decision Tree"]:
            base_model.set_params(class_weight=class_weight_dict)

        model = base_model
        model.fit(X_train_scaled, y_train_bal)

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        model_conf_matrices.append(cm)

        print("Confusion Matrix:")
        print(cm)

        print("Accuracy:", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(f1, 4))

        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)

        fold_num += 1
    total_cm = np.sum(model_conf_matrices, axis=0)

    print("\nSummed Confusion Matrix for", model_name)
    print(total_cm)

    final_results.append({
        "Model": model_name,
        "Mean Accuracy": np.mean(acc_scores),
        "Mean Precision": np.mean(prec_scores),
        "Mean Recall": np.mean(rec_scores),
        "Mean F1": np.mean(f1_scores)
    })

# ==========================
# FINAL TABLE
# ==========================
results_df = pd.DataFrame(final_results)
results_df = results_df.sort_values(by="Mean F1", ascending=False)

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")
print(results_df.to_string(index=False))