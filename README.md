 🎧 Audio Emotion Classification Web App

A deep learning-based web application for real-time **audio emotion recognition** using **MFCC features** and a trained neural network. Built with **Streamlit**, the app allows users to **upload or record audio**, classify emotional states, and view prediction probabilities.

---

## 🧠 Model Overview

This project uses a neural network trained on MFCC features extracted from audio signals to classify emotions. Key features include:

- **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction  
- **Trained deep learning model** using TensorFlow/Keras (`.h5` file)  
- Preprocessing with `StandardScaler` and `LabelEncoder`  
- Web interface built with **Streamlit**  
- Supports both **audio upload** and **recording**

---

## 🗂️ Repository Contents

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| `app.py`               | Streamlit web app source code                    |
| `model.h5`             | Trained Keras model                              |
| `scaler.pkl`           | Preprocessing scaler used during training        |
| `label_encoder.pkl`    | Encoder mapping for emotion classes              |
| `mars-open-project.ipynb` | Jupyter notebook with full training code     |
| `README.md`            | Project documentation                            |

---

## 🎯 Emotions Detected

- 😄 Happy  
- 😢 Sad  
- 😠 Angry  
- 😱 Fear  
- 😐 Neutral  

*(Adjust based on actual labels used in your model)*

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/audio-emotion-classifier.git
cd audio-emotion-classifier
2. Install Dependencies
Using requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install streamlit tensorflow librosa soundfile scikit-learn numpy
3. Run the App
bash
Copy
Edit
streamlit run app.py
📥 Audio Input Options
Upload a .wav audio file

Record audio directly in the browser (4 seconds max)

MFCC features are extracted and fed to the model for prediction

📊 Output
Probability scores for each emotion class

Final predicted emotion

📈 Model Training Details
See the mars-open-project.ipynb notebook for:

MFCC feature extraction pipeline

Model architecture and training

Evaluation metrics and visualizations

Saving/loading model and preprocessing objects

👤 Author
Luv .
Open Projects 2025 — Audio Emotion Recognition
GitHub
LinkedIn <!-- Optional -->
