# 🌦️ Weather Image Classification — TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-EfficientNetB0-red?logo=keras)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-brightgreen)

Klasyfikator zjawisk pogodowych na podstawie pojedynczego zdjęcia, zbudowany w
oparciu o **Transfer Learning**, **Fine Tuning** i **Grad-CAM** na
architekturze **EfficientNetB0** (TensorFlow / Keras). Projekt portfolio
prezentujący pełny, produkcyjny pipeline: od danych, przez trening, po
ewaluację, wyjaśnialność modelu i gotową funkcję predykcji na nowych
zdjęciach.

---

## 📖 Opis projektu

Model rozpoznaje **11 zjawisk pogodowych** na podstawie pojedynczego zdjęcia:
`dew` (rosa), `fogsmog` (mgła/smog), `frost` (szron), `glaze` (gołoledź),
`hail` (grad), `lightning` (błyskawica), `rain` (deszcz), `rainbow` (tęcza),
`rime` (szadź), `sandstorm` (burza piaskowa), `snow` (śnieg).

Celem projektu jest zaprezentowanie praktycznej, kompletnej znajomości
TensorFlow/Keras w zadaniu klasyfikacji obrazów: od eksploracji danych, przez
augmentację, dwufazowy trening (transfer learning + fine-tuning), pełną
ewaluację wieloma metrykami, aż po wyjaśnialność decyzji modelu (Grad-CAM) i
gotowy pipeline predykcji na nowych danych.

## 🧠 Architektura modelu

```
Input (224, 224, 3)
        ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness)
        ↓
EfficientNetB0 (bazowy, wagi ImageNet)
   Faza 1: w pełni zamrożony (transfer learning)
   Faza 2: odmrożone ostatnie 30 warstw, niski learning rate (fine-tuning)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (11, softmax)
```

**Strategia treningu — dwie fazy:**

| Faza | Model bazowy | Learning rate | Cel |
|---|---|---|---|
| 1. Transfer Learning | zamrożony | 1e-3 | trening głowicy klasyfikacyjnej |
| 2. Fine Tuning | ostatnie 30 warstw odmrożone | 1e-5 | precyzyjne dostrojenie cech wysokopoziomowych |

## 🛠️ Wykorzystane technologie

- **TensorFlow 2.x / Keras** — budowa i trening modelu
- **EfficientNetB0** (`tf.keras.applications`) — baza transfer learningu
- **Data Augmentation** — warstwy Keras (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomBrightness`)
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`
- **Grad-CAM** — wyjaśnialność decyzji modelu (Explainable AI)
- **scikit-learn** — precision, recall, F1-score, confusion matrix, classification report
- **Pandas / Matplotlib / Seaborn** — analiza danych i wizualizacje
- **Kaggle API** — pobieranie datasetu

## 📊 Wyniki

> Wartości metryk generowane są automatycznie w notebooku (sekcje 14–20) po
> przeprowadzeniu treningu i zależą od konkretnego przebiegu oraz sprzętu.
> Po uruchomieniu projektu warto wkleić tu finalne liczby, np.:

| Metryka | Wynik |
|---|---|
| Test Accuracy | _uzupełnij po treningu_ |
| Precision (macro) | _uzupełnij po treningu_ |
| Recall (macro) | _uzupełnij po treningu_ |
| F1-score (macro) | _uzupełnij po treningu_ |

Pełne wyniki (macierz pomyłek, raport klasyfikacji per klasa, krzywe
treningu) zapisywane są automatycznie do folderu `results/`:
- `results/metrics_summary.csv`
- `results/classification_report.csv`
- `results/confusion_matrix.png`
- `results/training_curves.png`

Zapisane predykcje na nowych zdjęciach (sekcja 27) trafiają do folderu
`predictions/`.

## 🖼️ Screenshoty

> Po uruchomieniu notebooka warto zapisać tu kluczowe wizualizacje (jako
> pliki w `images/`), np.:
> - przykładowe zdjęcie z każdej z 11 klas (sekcja 7),
> - efekt data augmentation (sekcja 8),
> - krzywe accuracy/loss (sekcja 13),
> - macierz pomyłek (sekcja 19),
> - wizualizacja Grad-CAM (sekcja 24) — to najbardziej efektowny screenshot do README.
>
> ```markdown
> ![Krzywe treningu](images/training_curves.png)
> ![Macierz pomyłek](images/confusion_matrix.png)
> ![Grad-CAM](images/gradcam_example.png)
> ```

## 🚀 Instrukcja uruchomienia

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/<twoj-uzytkownik>/weather-image-classification.git
cd weather-image-classification
```

### 2. Środowisko wirtualne i zależności

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Klucz API Kaggle (do pobrania datasetu)

1. Załóż darmowe konto na [kaggle.com](https://www.kaggle.com).
2. **Account → API → Create New API Token** — pobierze się plik `kaggle.json`.
3. Notebook zapyta o `KAGGLE_USERNAME` i `KAGGLE_KEY` interaktywnie
   (`getpass`) — wartości znajdziesz w pobranym pliku `kaggle.json`. Nie są
   one nigdzie zapisywane na stałe w kodzie.

### 4. Uruchomienie notebooka

```bash
jupyter notebook notebooks/Weather_Image_Classification_TensorFlow.ipynb
```

Notebook można też uruchomić bezpośrednio w **Google Colab** (zalecane —
darmowy dostęp do GPU): wystarczy wgrać plik `.ipynb` i ustawić
`Runtime → Change runtime type → GPU`.

### 5. Predykcja na własnym zdjęciu

Po wytrenowaniu (lub wczytaniu zapisanego) modelu:

```python
from tensorflow import keras
import json

model = keras.models.load_model("models/weather_classifier_efficientnetb0_final.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

# funkcja predict_new_image() zdefiniowana w notebooku (sekcja 27)
predict_new_image("images/moje_niebo.jpg", model, class_names)
```

## 📁 Struktura repozytorium

```
weather-image-classification/
├── notebooks/
│   └── Weather_Image_Classification_TensorFlow.ipynb   # główny notebook (30 sekcji)
├── models/
│   ├── phase1_feature_extraction.keras                 # checkpoint po fazie 1 (generowany)
│   ├── phase2_fine_tuning.keras                        # checkpoint po fazie 2 (generowany)
│   ├── weather_classifier_efficientnetb0_final.keras   # finalny model (generowany)
│   └── class_names.json                                # lista 11 nazw klas (generowany)
├── images/            # przykładowe zdjęcia / screenshoty do README
├── predictions/        # zapisane wyniki predykcji na nowych zdjęciach (generowane)
├── results/            # metryki, macierz pomyłek, raport klasyfikacji (generowane)
├── docs/                # dodatkowa dokumentacja
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🗺️ Roadmapa rozwoju

- [ ] Większe warianty EfficientNet (B3–B7) lub nowsze architektury (ConvNeXt, ViT)
- [ ] Połączenie z dodatkowymi zbiorami pogodowymi (WEAPD i inne) — większa i
      lepiej zbalansowana baza danych
- [ ] Wieloetykietowa klasyfikacja (współwystępowanie kilku zjawisk na zdjęciu)
- [ ] Analiza sekwencji czasowych z kamer monitoringu (CNN+LSTM, wideo-transformery)
- [ ] Eksport do TensorFlow Lite / ONNX pod urządzenia edge (stacje pogodowe)
- [ ] Wdrożenie jako REST API (FastAPI + Docker) z Grad-CAM w odpowiedzi API
- [ ] Aktywne uczenie (active learning) dla najbardziej niepewnych predykcji
- [ ] Monitoring modelu w produkcji (np. MLflow) — wykrywanie data driftu

Pełny opis ograniczeń modelu i możliwych kierunków rozwoju znajduje się w
sekcjach 29–30 notebooka.

## 📄 Licencja

Projekt udostępniony na licencji [MIT](LICENSE). Dataset Weather Dataset
podlega licencji określonej przez jego autora — patrz
[Kaggle — jehanbhathena/weather-dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset).

## 👤 Autor

Krzysztof — projekt portfolio przygotowany w celu zaprezentowania praktycznej
znajomości TensorFlow/Keras (Transfer Learning, Fine Tuning, Grad-CAM) w
zadaniach klasyfikacji obrazów.
