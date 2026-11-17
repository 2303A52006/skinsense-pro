# SkinSense Pro

**Pigmented skin lesion classifier with explainable AI. An educational project.**

**Live demo:** https://skinsense-pro.streamlit.app/
(Free hosting sleeps when idle, so the first load can take a minute.)

> **Disclaimer:** SkinSense is an educational tool, not a medical device. It cannot diagnose any condition. See a qualified dermatologist for any skin concern.

---

## What it does

Upload a photo of a skin spot. A ResNet-18 model ranks **7 types of pigmented skin lesions** by probability, shows how confident it is, and highlights the image areas that influenced the result (Grad-CAM). If the model is not confident enough, the app says **Uncertain** instead of guessing and advises seeing a dermatologist.

### The 7 classes

| Code | Condition | Severity label in the app |
|---|---|---|
| `nv` | Melanocytic Nevus (common mole) | Low |
| `mel` | Melanoma | High |
| `bcc` | Basal Cell Carcinoma | High |
| `akiec` | Actinic Keratosis | Medium |
| `bkl` | Benign Keratosis | Low |
| `df` | Dermatofibroma | Low |
| `vasc` | Vascular Lesion | Low |

**Not covered:** rashes, fungal and other infections, acne, vitiligo, and other skin cancers such as squamous cell carcinoma. For these, the model can only guess among its 7 classes, so results will be wrong or Uncertain.

## Features

- **Analysis:** top-3 possibilities, full probability chart, Grad-CAM heatmap, adjustable low-confidence threshold
- **Crop tool** to focus on the lesion, and an **image quality check** (size, blur, lighting)
- **Skin journal:** save results with notes and a body site, edit or delete entries
- **Lesion timeline:** follow the same body site over time with a confidence chart and photo strip
- **Compare** two saved analyses side by side
- **Dashboard** with disease distribution, severity levels and confidence over time
- **PDF reports** with the image, Grad-CAM, result and disclaimer
- **Model Evaluation page:** accuracy, per-class recall, confusion matrix, calibration and threshold analysis
- **Educational library:** what the model can and cannot detect, plus the ABCDE rule for moles
- **Settings:** dark and light themes, font size, display name, CSV export, clear history

## Model

- **Data:** [HAM10000](https://doi.org/10.1038/sdata.2018.161), about 10,000 dermatoscopic images in 7 classes
- **Split:** 80/20 **grouped by `lesion_id`**, so the same lesion never appears in both training and validation (7,991 training and 2,024 validation images)
- **Network:** ResNet-18, ImageNet pre-trained, fine-tuned (transfer learning)
- **Final training (v4):** 12 epochs, AdamW (lr 2e-4, cosine schedule), class-weighted cross-entropy with extra weight (1.5x) on melanoma and basal cell carcinoma, augmentation (flips, rotation, zoom crops, colour jitter, slight blur), final epoch kept
- **Training script:** [`training/train_v4.py`](training/train_v4.py) (run in Google Colab)

### Results on the validation set

| Metric | First model | Final model (v4) |
|---|---|---|
| Accuracy | 68.6% | **74.2%** |
| Balanced accuracy | **72.5%** | 68.8% |
| Melanoma recall | 64.0% | **83.5%** |
| Basal cell carcinoma recall | 54.4% | **78.9%** |
| Calibration error (ECE) | **6.8** | 10.2 |

Recall per class:

| Condition | First model | Final model (v4) |
|---|---|---|
| Melanoma | 64.0% | **83.5%** |
| Basal Cell Carcinoma | 54.4% | **78.9%** |
| Melanocytic Nevus | 69.8% | **74.9%** |
| Benign Keratosis | 66.3% | 67.1% |
| Vascular Lesion | **91.2%** | 82.3% |
| Actinic Keratosis | **73.1%** | 50.8% |
| Dermatofibroma | **88.9%** | 44.4% |

**Trade-off:** the final model catches many more cancers (melanoma and basal cell carcinoma) but is weaker on some rarer, milder classes, and its confidence scores are less well calibrated. Moles make up about two thirds of the data, so plain accuracy is misleading. Balanced accuracy and per-class recall are more informative. Dermatofibroma (18 validation images) and vascular lesion (34) are too small for reliable percentages.

## Limitations

- The model sees **7 pigmented lesion classes only**. It does not cover rashes, infections or other skin diseases.
- Trained on **dermoscopy images**. Ordinary phone photos are a different domain, so results on them are less reliable.
- There is **no separate test set**. Hyperparameters were chosen after looking at validation results, so the numbers are slightly optimistic.
- Small classes make per-class numbers noisy.
- The model is **not clinically validated** and must not be used for diagnosis.

## Privacy

When deployed with `SKINSENSE_PUBLIC=1`, every visitor gets a private session folder. Photos and journals are not shared between visitors and are deleted automatically after a few hours of inactivity. When run locally, data is stored in the `data/` folder on your own computer.

## Screenshots

| Welcome | New analysis |
|---|---|
| ![Welcome](assets/welcome.png) | ![New analysis](assets/new-analysis.png) |

| Model evaluation | Lesion timeline |
|---|---|
| ![Model evaluation](assets/model-evaluation.png) | ![Lesion timeline](assets/timeline.png) |

## Run it locally

Tested with Python 3.13.

```bash
git clone https://github.com/2303A52006/skinsense-pro.git
cd skinsense-pro
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at http://localhost:8501. On macOS or Linux, if installing `torch` fails, remove the `+cpu` from the `torch` and `torchvision` lines (or install PyTorch following pytorch.org).

## Project structure

```
app.py                      Streamlit application
requirements.txt            Python packages (CPU-only PyTorch)
packages.txt                System library for OpenCV on the server
models/skinsense_model.pth  Trained model weights
data/eval_results.json      Validation results shown on the Model Evaluation page
training/train_v4.py        Training and evaluation script (Colab)
.streamlit/config.toml      Streamlit settings
```

## Tech stack

Python, Streamlit, PyTorch, torchvision, pytorch-grad-cam, OpenCV, SQLite, pandas, Plotly, ReportLab.

## Author

SAI SNEHITH