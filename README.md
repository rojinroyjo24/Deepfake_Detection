# 🎭 Deepfake Video Detection

A deep learning-based web application that detects whether a video is **FAKE** (deepfake) or **REAL** using a **ResNeXt + LSTM** architecture.

---

## 🧠 How It Works

```
Video Upload → Frame Extraction → Face Detection → Feature Extraction → LSTM Classification
```

1. **Frame Extraction** — Samples frames from the uploaded video
2. **Face Detection** — Detects and crops faces from each frame using OpenCV
3. **Feature Extraction** — Passes face frames through a pre-trained ResNeXt-50 CNN to extract 2048-dimensional features
4. **Classification** — An LSTM processes the temporal sequence of features and predicts FAKE or REAL with confidence

---

## 🏗️ Project Structure

```
Deepfake_video_Detection/
├── app.py                  # Flask web server (entry point)
├── train.py                # Model training script
├── predict_video.py        # Prediction pipeline
├── preprocess.py           # Data preprocessing
├── database.py             # SQLite prediction history
├── test.py                 # Test script
├── requirements.txt        # Python dependencies
├── best_model.pth          # Trained model weights
├── models/
│   └── resnext_lstm.py     # ResNeXt + LSTM architecture
├── utils/
│   ├── face_extraction.py  # Face detection from frames
│   ├── feature_extraction.py  # ResNeXt feature extraction
│   └── video_processing.py    # Frame extraction from video
├── templates/
│   └── index.html          # Web interface
└── test_videos/            # Sample demo videos (small clips)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/rojinroyjo24/Deepfake_Detection.git
cd Deepfake_Detection
```

### 2. Create a virtual environment
```bash
python -m venv env
env\Scripts\activate        # Windows
# or
source env/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

Then open your browser at: `http://127.0.0.1:5000`

---

## 🎬 Demo

Upload any `.mp4` video through the web interface. The system will:
- Analyze the video frames
- Detect faces in each frame
- Output a **FAKE / REAL** prediction with a confidence probability

Sample videos for testing are included in the `test_videos/` directory.

---

## 🧪 Training Your Own Model

The model is trained on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset.

### Preprocess data
```bash
python preprocess.py
```

### Train the model
```bash
python train.py
```

The best model is saved as `best_model.pth` automatically.

---

## 🤖 Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | ResNeXt-50 (pre-trained on ImageNet) |
| Temporal | LSTM (hidden size: 256) |
| Dropout | 0.5 |
| Output | Sigmoid (binary classification) |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr=0.0005) |

---

## 📦 Dependencies

- Python 3.8+
- PyTorch 2.x
- OpenCV
- Flask
- NumPy, Pandas
- torchvision

See `requirements.txt` for full dependency list.

---

## ⚠️ Note on Large Files

- The `data/` folder (training dataset) is **not included** in this repo. Download the FaceForensics++ dataset separately.
- `processed_data/` and `extracted_frames/` are generated locally during preprocessing — not committed.
- `best_model.pth` (~9 MB) is included for immediate inference without retraining.

---

## 📄 License

This project is for academic and research purposes.

---

## 👤 Author

**Rojin Roy** — [@rojinroyjo24](https://github.com/rojinroyjo24)
