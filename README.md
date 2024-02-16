This project combines two powerful techniques - Text-based Emotion Recognition using BERT Preprocessor and TensorFlow library , and Speech/Audio-based Emotion Recognition using a Random Forest Classifier ML model and Convulational Neural Networks (CNN).

*Overview*
This repository contains code for recognizing emotions from both text and speech data. The primary aim is to demonstrate how various machine learning techniques can be applied to recognize emotions from different modalities.

*Dependencies*
The following dependencies have to be installed:

TensorFlow
Hugging Face Transformers
Pandas
NumPy
Scikit-learn
Librosa (for audio processing)

*Text Emotion Recognition (BERT Preprocessor)*
The text-based emotion recognition module utilizes the BERT preprocessor for encoding text data. The steps involved are:

Data Preprocessing: Preprocess the text data to prepare it for BERT encoding.
BERT Encoding: Use the BERT preprocessor to convert text inputs into BERT embeddings.
Model Training: Train a TensorFlow model on the encoded text data for emotion recognition.

*Speech/Audio Emotion Recognition (Random Forest Classifier)*
The speech/audio-based emotion recognition module employs a Random Forest Classifier model and Convulational Neural Networks trained on various features extracted from audio data. The process includes:

Feature Extraction: Extract relevant features from the audio files using Librosa.
Data Preparation: Prepare the extracted features along with corresponding labels for model training.
Model Training: Train a Random Forest Classifier model and Convulational Neural Networks (CNN) on the extracted features.
