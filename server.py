from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("final_model.pkl")
scaler = joblib.load("final_scaler.pkl")
word_references = np.load("word_references.npy", allow_pickle=True).item()
print(word_references.keys())

MAX_FRAMES = 40

# =================================
# FEATURE EXTRACTION
# =================================
def extract_features(file_path):
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

    mfcc = np.nan_to_num(mfcc)

    flat = mfcc.reshape(1, -1)
    scaled = scaler.transform(flat)

    return scaled, audio_trimmed


# =================================
# BASIC SPEECH CHECK
# =================================
def is_valid_speech(audio):
    duration = len(audio) / 16000
    rms = np.mean(librosa.feature.rms(y=audio))

    print("Duration:", duration)
    print("RMS:", rms)

    if duration < 0.20:
        return False

    if rms < 0.010:
        return False

    return True

# =================================
# WORD MATCH CHECK
# =================================
def compare_to_target_word(feature_vector, word_id):
    print("Looking for word id:", word_id)
    print("Available ids:", word_references.keys())

    if word_id not in word_references:
        return 9999

    reference = word_references[word_id]
    distance = np.linalg.norm(feature_vector[0] - reference)

    print("Distance to word", word_id, "=", distance)
    return distance


# =================================
# API
# =================================
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    word_id = str(int(float(request.form['word_id'])))

    temp_path = "temp_audio.wav"
    file.save(temp_path)

    print("Saved file size:", os.path.getsize(temp_path))
    print("Current Word ID:", word_id)

    features, trimmed_audio = extract_features(temp_path)

    # stage 1
    if not is_valid_speech(trimmed_audio):
        return jsonify({
            "prediction": "Incorrect",
            "correct_probability": 0.0,
            "therapy_level": "retry"
        })

    # stage 2
    distance = compare_to_target_word(features, word_id)

    if distance <= 120:
        return jsonify({
            "prediction": "Correct",
            "correct_probability": 0.95,
            "therapy_level": "excellent"
        })

    elif distance <= 220:
        return jsonify({
            "prediction": "Correct",
            "correct_probability": 0.70,
            "therapy_level": "good_try"
        })

    else:
        return jsonify({
            "prediction": "Incorrect",
            "correct_probability": 0.20,
            "therapy_level": "retry"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051)