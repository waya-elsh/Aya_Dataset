from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np

app = Flask(__name__)

model = joblib.load("final_model.pkl")
scaler = joblib.load("final_scaler.pkl")

MAX_FRAMES = 40


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # EXACT SAME AS TRAINING
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

    mfcc = np.nan_to_num(
        mfcc,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    flat = mfcc.reshape(1, -1)
    scaled = scaler.transform(flat)

    return scaled


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']

    temp_path = "temp_audio.wav"
    file.save(temp_path)

    features = extract_features(temp_path)

    probs = model.predict_proba(features)[0]
    print("Raw probabilities:", probs)

    print("Raw probabilities:", probs)

    return jsonify({
    "prediction": "Correct" if probs[1] >= 0.5 else "Incorrect",
    "correct_probability": float(probs[1])
})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)