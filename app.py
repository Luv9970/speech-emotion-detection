import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model
import io
import joblib
# Load model and preprocessing objects
model = load_model("emotion_lstm_model.h5")
scaler = joblib.load("emotion_scaler.pkl")
label_encoder = joblib.load("label_encoder_lstm.pkl")
class_labels = label_encoder.classes_

st.title("🎤 Real-time Audio Emotion Classifier")

# --- Upload or record audio ---
audio_source = st.radio("Choose audio input method:", ["Upload", "Record"])

audio_bytes = None
if audio_source == "Upload":
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
else:
    st.info("Record for 4 seconds")
    audio_bytes = st.audio_recorder("Click to record", format="audio/wav")

# --- Feature extraction ---
def extract_mfcc(audio_bytes):
    try:
        # Load from bytes buffer
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if y.ndim > 1:  # Stereo to mono
            y = np.mean(y, axis=1)
        y = y[:sr * 4]  # Trim to 4 seconds
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None

# --- Predict emotion ---
if audio_bytes and st.button("🎯 Predict Emotion"):
    mfcc_features = extract_mfcc(audio_bytes)
    if mfcc_features is not None:
        input_scaled = scaler.transform([mfcc_features])
        input_reshaped = input_scaled.reshape(1, 1, -1)  # (batch, timesteps, features)
        probs = model.predict(input_reshaped)[0]

        # Display probabilities
        st.subheader("Emotion Probabilities")
        for label, prob in zip(class_labels, probs):
            st.write(f"**{label}**: {prob:.2%}")

        # Show prediction
        predicted_emotion = class_labels[np.argmax(probs)]
        st.success(f"🎧 Predicted Emotion: **{predicted_emotion}**")
