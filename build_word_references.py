import pandas as pd
import librosa
import numpy as np
import joblib

CSV_PATH = "MetadataAya.csv"
MAX_FRAMES = 40

metadata = pd.read_csv(CSV_PATH, sep=';')
metadata.columns = metadata.iloc[0]
metadata = metadata[1:].reset_index(drop=True)
metadata.columns = metadata.columns.str.strip()

word_vectors = {}

# نستخدم نفس scaler الذي تدرب به الموديل
scaler = joblib.load("final_scaler.pkl")

for _, row in metadata.iterrows():
    file_path = row["File path"]
    label = str(row["Label"]).strip().lower()
    word_id = str(row["Word id"]).strip()

    # ناخذ فقط العينات الصحيحة كمرجع
    if label != "correct":
        continue

    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

        mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)

        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_FRAMES]

        mfcc = np.nan_to_num(mfcc)

        flat = mfcc.reshape(1, -1)
        scaled = scaler.transform(flat)[0]

        if word_id not in word_vectors:
            word_vectors[word_id] = []

        word_vectors[word_id].append(scaled)

    except Exception as e:
        print("Skipped:", file_path, e)

# نحسب center لكل كلمة
word_references = {}

for word_id, vectors in word_vectors.items():
    center = np.mean(vectors, axis=0)
    word_references[word_id] = center

print("Built references for", len(word_references), "words")

np.save("word_references.npy", word_references, allow_pickle=True)
print("Saved word_references.npy successfully")