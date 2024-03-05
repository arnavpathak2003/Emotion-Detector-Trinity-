import streamlit as st
import numpy as np
import pickle
import librosa
import io


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


st.set_page_config(
    page_title="Emotion Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    "<h1 style='text-align: center;'>Emotion Detector</h1>", unsafe_allow_html=True
)

st.text(
    """Please enter your text or audio and the model will predict your emotion ! """
)

col1, col2 = st.columns([3, 6])

with col1:
    text_input = st.text_input("Enter some text:")
    if text_input:
        import tensorflow as tf
        import tensorflow_text as text

with col2:
    audio_input = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if st.button("Predict"):
    if text_input:
        from tensorflow.keras.models import load_model

        text_emotion_model = load_model("text_emotion")
        # prediction = text_emotion_model.predict(
        #     ["Remember what she said in my last letter? "]
        # )
        prediction = text_emotion_model.predict([text_input])
        V = prediction[0][0] / 5 * 100
        A = prediction[0][1] / 5 * 100
        D = prediction[0][2] / 5 * 100

        st.write(f"Valance = {V} %")
        st.write(f"Arousal = {A} %")
        st.write(f"Dominance = {D} %")
    elif audio_input:
        with open("audio_emotion.pkl", "rb") as file:
            audio_bytes = audio_input.getvalue()

            audio_file = io.BytesIO(audio_bytes)

            data, sample_rate = librosa.load(audio_file, sr=None)
            features = extract_features(data, sample_rate)

            audio_emotion_model = pickle.load(file)

            prediction = audio_emotion_model.predict(features.reshape(1, -1))[0]
            st.write(f"The emotion of the speaker is {prediction}.")

    else:
        st.write("Please upload a file first...")
