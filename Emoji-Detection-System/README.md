# 😀 Emoji Detection System

A simple emoji/emotion detection pipeline that preprocesses facial images from the FER-2013 dataset, trains a classifier, and overlays predicted emojis on a live webcam feed.

## 📋 Table of Contents

1. [🚀 Overview](#-overview)
2. [✨ Features](#-features)
3. [🛠️ Getting Started](#-getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [▶️ Usage](#usage)

   * [Data Preparation](#data-preparation)
   * [Preprocessing](#preprocessing)
   * [Training](#training)
   * [Inference](#inference)
5. [📁 Project Structure](#-project-structure)
6. [⚙️ Configuration](#-configuration)
7. [🤝 Contributing](#-contributing)
8. [📄 License](#-license)
9. [📫 Contact](#-contact)

## 🚀 Overview

A quick description of your pipeline:

* **Input:** FER-2013 facial images or live webcam feed
* **Output:** Real-time emoji overlays matching detected emotions
* **Purpose:** Demonstrate facial emotion recognition with emoji visualization

## ✨ Features

* 🔍 **Face detection & cropping** via OpenCV Haar cascades
* 🖼️ **Bilateral filtering** for noise reduction
* 🔄 **Resize & normalization** to 64×64 grayscale images
* 🏷️ **One-hot encoding** of emotion labels
* 📊 **Train/validation split** for hyperparameter tuning
* 🎥 **Real-time emoji overlay** on live webcam feed

## 🛠️ Getting Started

### Prerequisites

* Python 3.8+ installed
* OpenCV (`opencv-python`)
* NumPy, scikit-learn
* (Optional) Git LFS for large datasets

### Installation

```bash
git clone https://github.com/<YourUser>/Emoji-Detection-System.git
cd Emoji-Detection-System
pip install -r requirements.txt
```

## ▶️ Usage

### Data Preparation

1. Download and extract the `FER-2013_sampled` archive into the project root.
2. Ensure it contains two subfolders:

   * `train_balanced_7000/`
   * `test_balanced_7000/`

### Preprocessing

Run the Jupyter notebook or script to generate `.npy` files:

```bash
cd Notebooks
jupyter notebook preprocessing.ipynb
# or
python ../scripts/preprocess_data.py
```

### Training

Train the model (example using scikit-learn MLP):

```bash
python train.py --data-dir FER-2013_sampled --output-model models/emoji_mlp.pkl
```

### Inference

Launch the live webcam demo:

```bash
python inference.py --model models/emoji_mlp.pkl
```

## 📁 Project Structure

```
Emoji-Detection-System/
├── FER-2013_sampled/            # Sampled dataset splits
│   ├── train_balanced_7000/     # Training images organized by emotion
│   └── test_balanced_7000/      # Test images organized by emotion
├── Notebooks/                   # Jupyter notebooks
│   └── preprocessing.ipynb      # Data loading & preprocessing
├── scripts/                     # Standalone scripts
│   └── preprocess_data.py
├── models/                      # Saved model files
│   └── emoji_mlp.pkl
├── templates/                   # Emoji PNG assets with transparency
├── inference.py                 # Real-time emoji overlay script
├── train.py                     # Training entrypoint
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## ⚙️ Configuration

Optional flags or environment variables:

```bash
python train.py --epochs 30 --batch-size 64
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

## 📄 License

MIT © 2025 Your Name

## 📫 Contact

Your Name — [your.email@example.com](mailto:your.email@example.com)
Project Link: [https://github.com/](https://github.com/)<YourUser>/Emoji-Detection-System
