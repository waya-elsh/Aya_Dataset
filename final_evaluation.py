import joblib
import pandas as pd
import librosa
import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

CSV_PATH = "MetadataAya.csv"
MAX_FRAMES = 40
RANDOM_STATE = 42
RATIO_CORRECT_TO_INCORRECT = 3


metadata = pd.read_csv(CSV_PATH, sep=';')
metadata.columns = metadata.iloc[0]
metadata = metadata[1:].reset_index(drop=True)
metadata.columns = metadata.columns.str.strip()

X_list = []
y_list = []


for  row in metadata.iterrows():
    file_path = row["File path"]
    label = row["Label"]

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

    except Exception as e:
        print("Skipped:", file_path)
        print(e)


X = np.array(X_list)
y = np.array([label.strip().lower() for label in y_list])

label_map = {
    "correct": 1,
    "incorrect": 0
}

y_encoded = np.array([label_map[label] for label in y])

print("Original distribution:", Counter(y_encoded))



np.random.seed(RANDOM_STATE)

correct_idx = np.where(y_encoded == 1)[0]
incorrect_idx = np.where(y_encoded == 0)[0]

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

final_idx = np.concatenate([correct_sampled, incorrect_sampled])
np.random.shuffle(final_idx)

X_final = X[final_idx]
y_final = y_encoded[final_idx]

print("Final training distribution:", Counter(y_final))


X_final_flat = X_final.reshape(X_final.shape[0], -1)



scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final_flat)

X_final_scaled = np.nan_to_num(
    X_final_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


classes = np.unique(y_final)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_final
)

class_weight_dict = dict(zip(classes, class_weights))



model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight=class_weight_dict,
    random_state=RANDOM_STATE
)

model.fit(X_final_scaled, y_final)



joblib.dump(model, "final_model.pkl")
joblib.dump(scaler, "final_scaler.pkl")

print("\nFinal deployment model saved successfully.")
print("- final_model.pkl")
print("- final_scaler.pkl")