# 🌦️ Weather Image Classification — TensorFlow

A weather-phenomenon classifier based on a single photograph, built on
**Transfer Learning**, **Fine Tuning**, and **Grad-CAM** on the
**EfficientNetB0** architecture (TensorFlow / Keras). A project
presenting a full, production-style pipeline: from data, through training, to
evaluation, model explainability, and a ready-to-use prediction function for
new images.

---

## 📖 Project Description

The model recognizes **11 weather phenomena** from a single photograph:
`dew`, `fogsmog`, `frost`, `glaze`, `hail`, `lightning`, `rain`, `rainbow`,
`rime`, `sandstorm`, `snow`.

The goal of this project is to demonstrate how TensorFlow/Keras can be used for image classification by covering the entire workflow—from data exploration and augmentation, through two-stage training (transfer learning followed by fine-tuning), comprehensive evaluation using multiple performance metrics, to model explainability with Grad-CAM and a production-ready prediction pipeline for new data.

## 🧠 Model Architecture

```
Input (224, 224, 3)
        ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness)
        ↓
EfficientNetB0 (base, ImageNet weights)
   Phase 1: fully frozen (transfer learning)
   Phase 2: last 30 layers unfrozen, low learning rate (fine-tuning)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (11, softmax)
```

**Training strategy — two phases:**

| Phase | Base model | Learning rate | Goal |
|---|---|---|---|
| 1. Transfer Learning | frozen | 1e-3 | training the classification head |
| 2. Fine Tuning | last 30 layers unfrozen | 1e-5 | precise adjustment of high-level features |

## 🛠️ Technologies Used

- **TensorFlow 2.x / Keras** — model building and training
- **EfficientNetB0** (`tf.keras.applications`) — transfer learning base
- **Data Augmentation** — Keras layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomBrightness`)
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`
- **Grad-CAM** — model decision explainability (Explainable AI)
- **scikit-learn** — precision, recall, F1-score, confusion matrix, classification report
- **Pandas / Matplotlib / Seaborn** — data analysis and visualization
- **Kaggle API** — dataset download

## 📊 Results

> Metric values are generated automatically in the notebook (Sections 14–20)
> after training and depend on the specific training run and hardware.

| Metric |
|---|
| Test Accuracy |
| Precision (macro) |
| Recall (macro) |
| F1-score (macro) |

Full results (confusion matrix, per-class classification report, training
curves) are automatically saved to the `results/` folder:
- `results/metrics_summary.csv`
- `results/classification_report.csv`
- `results/confusion_matrix.png`
- `results/training_curves.png`

Saved predictions on new images (Section 27) go to the `predictions/` folder.

## 🖼️ Screenshots

> After running the notebook, it's worth saving key visualizations here (as
> files in `images/`), e.g.:
> - an example image from each of the 11 classes (Section 7),
> - the data augmentation effect (Section 8),
> - accuracy/loss curves (Section 13),
> - the confusion matrix (Section 19),
> - the Grad-CAM visualization (Section 24) — the most striking screenshot
>   for the README.
>
> ```markdown
> ![Training curves](images/training_curves.png)
> ![Confusion matrix](images/confusion_matrix.png)
> ![Grad-CAM](images/gradcam_example.png)
> ```

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Weather-Image-Classification.git
cd weather-image-classification
```

### 2. Virtual environment and dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Kaggle API key (to download the dataset)

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. **Account → API → Create New API Token** — this downloads a `kaggle.json`
   file.
3. The notebook will interactively prompt for `KAGGLE_USERNAME` and
   `KAGGLE_KEY` (`getpass`) — you'll find the values in the downloaded
   `kaggle.json` file. They are never hard-coded anywhere.

### 4. Run the notebook

```bash
jupyter notebook notebooks/Weather_Image_Classification_TensorFlow.ipynb
```

The notebook can also be run directly in **Google Colab** (recommended —
free GPU access): just upload the `.ipynb` file and set
`Runtime → Change runtime type → GPU`.

### 5. Predict on your own image

After training (or loading a saved model):

```python
from tensorflow import keras
import json

model = keras.models.load_model("models/weather_classifier_efficientnetb0_final.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

# predict_new_image() function defined in the notebook (Section 27)
predict_new_image("images/my_sky.jpg", model, class_names)
```

## 📁 Repository Structure

```
Weather-Image-Classification/
├── notebooks/
│   └── Weather_Image_Classification_TensorFlow.ipynb 
├── models/             
├── images/           
├── predictions/        
├── results/           
├── docs/               
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🗺️ Development Roadmap

- [ ] Larger EfficientNet variants (B3–B7) or newer architectures (ConvNeXt, ViT)
- [ ] Combine with additional weather datasets (WEAPD and others) — a larger,
      better-balanced data pool
- [ ] Multi-label classification (co-occurrence of several phenomena in one image)
- [ ] Temporal sequence analysis from monitoring cameras (CNN+LSTM, video transformers)
- [ ] Export to TensorFlow Lite / ONNX for edge devices (weather stations)
- [ ] Deployment as a REST API (FastAPI + Docker) with Grad-CAM in the API response
- [ ] Active learning for the least confident predictions
- [ ] Production model monitoring (e.g. MLflow) — data drift detection

A full description of the model's limitations and possible development
directions can be found in Sections 29–30 of the notebook.

## 📄 License

This project is released under the [MIT](LICENSE) license. The Weather
Dataset is subject to the license set by its author — see
[Kaggle — jehanbhathena/weather-dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset).

## 👤 Author

AIResearchForge — a TensorFlow/Keras-based project for image classification, featuring Transfer Learning, Fine-Tuning, and model explainability using Grad-CAM.
