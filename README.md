# Handwritten Digit Recognition

A beginner-friendly computer vision project that recognizes handwritten digits (0–9) from
images. It combines neural networks trained on the MNIST dataset (a baseline MLP and a
convolutional network) with a classical OpenCV preprocessing pipeline, so it works on real
photos of digits written on paper — not just clean benchmark images. One command starts an
interactive menu where you can train a model or predict digits from a folder of images.

## Key Features

- Interactive terminal menu with two workflows: **Learn / Train** and **Predict**
- Two trainable models: a baseline MLP and a CNN with early stopping
- Batch prediction over every image in `images/` (PNG, JPG, JPEG, BMP), sorted by filename
- Per-image error handling — unreadable images are skipped and reported, the run continues
- Automatic background inversion, contour cropping, and MNIST-style centering of real-digit photos
- Friendly messages for missing dataset, missing model, or empty image folder
- Runs from any working directory; trained weights and the dataset are included

## Technologies

| Technology | Role |
| --- | --- |
| Python 3.12 | Programming language (verified with Python 3.12.6) |
| TensorFlow / Keras 2.19 | Model definition, training, and inference |
| OpenCV (`opencv-python`) 4.10 | Image loading and classical preprocessing |
| NumPy 2.2 | Dataset loading and array operations |

## Project Structure

```text
.
├── app.py                  # Single entry point (interactive menu)
├── src/
│   ├── training.py         # Dataset loading + MLP and CNN training pipelines
│   └── prediction.py       # Image discovery, OpenCV preprocessing, batch prediction
├── images/                 # Prediction input folder (sample digit images included)
├── models/                 # Trained model weights (.h5)
├── datasets/
│   └── mnist.npz           # MNIST dataset used for training
├── docs/
│   └── linkedin-post.md    # Optional LinkedIn post draft
├── requirements.txt
└── .gitignore
```

## Supported Models

Both models are Keras `Sequential` classifiers, compiled with `adam` and
`sparse_categorical_crossentropy` (integer labels, no one-hot encoding needed), and both
output 10 softmax probabilities.

| | MLP (baseline) | CNN (convolutional) |
| --- | --- | --- |
| Input handling | `Flatten` → 784-vector | `Reshape` → (28, 28, 1) |
| Feature layers | 2 × Dense(128, ReLU) | Conv2D 32 → MaxPool → Conv2D 64 → MaxPool → Conv2D 128 (all 3×3, ReLU, `padding="same"`) |
| Regularization | Dropout 0.2 | Dropout 0.3 + early stopping |
| Default epochs | 5 | up to 20 |
| Validation | none | `validation_split=0.2` |
| Checkpoint | `models/mnist_model.h5` | `models/my_model.h5` |

The essential difference: the MLP discards spatial structure (every pixel connects to every
neuron), while the CNN uses local receptive fields and shared weights, building a hierarchy
of features from edges to digit shapes. The CNN additionally uses
`EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`, so training
stops when `val_loss` stops improving and the best weights are restored — the effective
epoch count can be lower than requested.

`models/my_model_2.h5` is a second, separate training run of the same CNN architecture; it
is offered as an extra choice during prediction. All three checkpoints were trained
identically on normalized MNIST input, which is why they share one preprocessing pipeline.

## Dataset and Training Procedure

**MNIST** — 70,000 grayscale images of handwritten digits (28×28 pixels, uint8), split into
**60,000 training and 10,000 test images**, labels 0–9. The project loads it from the local
archive `datasets/mnist.npz` (keys: `x_train`, `y_train`, `x_test`, `y_test`) rather than
downloading at runtime. A copy is committed, so no download is needed.

Training steps (identical for both models):

1. Load `datasets/mnist.npz`.
2. Normalize pixels by dividing by `255.0`, mapping them to `[0, 1]`. No augmentation or
   further normalization.
3. Train with the settings in the table above (`model.fit`).
4. Evaluate on the 10,000 test images and print the test accuracy of your run.
5. Save the model in the legacy Keras `.h5` format into `models/`.

No accuracy results are stored or claimed by this repository — each run prints its own.

## Real-Image Preprocessing Pipeline (OpenCV)

MNIST digits are centered, 28×28, and stroke-normalized; phone photos are not. For each
prediction image, this pipeline runs before the network sees it:

| Step | Operation | Why |
| --- | --- | --- |
| 1 | `cv2.imread(..., IMREAD_GRAYSCALE)` | The network expects single-channel input |
| 2 | `cv2.GaussianBlur(image, (5, 5), 0)` | Suppress camera noise and paper texture |
| 3 | If `np.mean(image) > 127`: `cv2.bitwise_not(image)` | Bright-background photos are inverted so the digit becomes the bright region, matching MNIST |
| 4 | `cv2.threshold(image, 120, 255, THRESH_BINARY)` | Remove gray levels so the digit becomes a solid shape |
| 5 | `cv2.findContours(RETR_EXTERNAL, ...)`, keep the largest contour, crop to its bounding rect | Isolate the digit and drop surrounding whitespace |
| 6 | Resize the crop to fit inside 20×20, preserving aspect ratio | Matches the classic LeCun-style "fit in 20×20" normalization |
| 7 | Center the result on a 28×28 black canvas | Restores the exact input size while keeping the digit centered |
| 8 | Divide by 255.0, reshape to (1, 28, 28), predict | Same normalization as training |
| 9 | `argmax` over the 10 softmax outputs | Final predicted digit + confidence |

Prediction is **batch**: all supported files directly inside `images/` (`.png`, `.jpg`,
`.jpeg`, `.bmp`) are processed in sorted filename order. Subdirectories and unsupported
files are ignored. Unreadable images or images without any contour are reported as failures
and do not stop the run.

## Installation and Setup

Prerequisites: Python 3.10+ (verified with 3.12.6 on Windows) and ~1 GB of free disk space
for TensorFlow.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

```text
=================================
      Computer Vision Project
=================================

1. Learn / Train a Model
2. Predict Digits from Images
3. Exit

Choose an option:
```

Invalid input is rejected politely, and the app returns to the main menu after each
workflow. Nothing is trained or predicted until you choose an option.

### Training a model

1. Choose option `1`.
2. Pick a model: `1. MLP (baseline)` or `2. CNN (convolutional)`.
3. Press Enter to accept the default number of epochs (5 for the MLP, 20 for the CNN), or
   type a positive whole number.
4. Watch the training progress; the final test accuracy is printed and the model is saved
   into `models/`.

```text
Starting training: CNN (convolutional) for 20 epoch(s)...
...
Test accuracy: 0.9...
Done. Model saved to: models\my_model.h5
```

*(Accuracy value shown as a placeholder — your run prints its own result.)*

If `datasets/mnist.npz` is missing, the app explains how to supply it and returns to the
menu. You can also replace any checkpoint by copying your own `.h5` file into `models/`.

### Predicting digits from `images/`

1. Put one or more digit images into the `images/` folder (sample images are included).
2. Choose option `2`.
3. If several trained models exist, pick one; with a single model it is used automatically.
4. Read the predictions, then the summary. The app returns to the main menu.

## Example Terminal Output

**Illustrative — the numbers below show the output format only, not actual results:**

```text
Predictions
-------------------------
test-digit.png -> 2 (confidence 0.9876)
test-digit2.png -> 5 (confidence 0.9512)
sample-03.jpg -> 8 (confidence 0.8934)

Successfully processed: 3
Failed: 0
```

With an unreadable file:

```text
bad.png -> skipped (could not read image)
...
Successfully processed: 7
Failed: 1
Failed files: bad.png
```

## Limitations

- **One digit per image.** There is no multi-digit OCR or segmentation of several digits in
  one photo; the pipeline keeps the largest contour only.
- **Photos should show a single, reasonably clean digit.** Large shadows, borders, or extra
  objects can confuse the contour-cropping step.
- **Low confidence means a poor match.** Confidence values well below 1.0 (sometimes under
  0.5) are common on photos that differ from MNIST's clean, centered style — treat them as
  uncertainty, not as a correct prediction.
- **No benchmark results are claimed.** Test accuracy is printed at runtime by your own
  training run; the repository stores no metrics.
- **Unseeded training.** Results vary slightly between runs.
- **Minor numeric non-determinism at inference.** TensorFlow's oneDNN optimizations can
  produce slightly different floating-point results between runs, which can flip the
  prediction of borderline images.
- **Legacy `.h5` model format.** Newer Keras versions may print deprecation warnings when
  loading models.
- **Threshold 120 and blur 5×5 were tuned empirically** against the included sample photos;
  unusual lighting may need different values.
- Comments inside `src/` are partly in Persian (Farsi).

## Future Improvements

Not implemented — ideas only:

- Center-of-mass shifting and data augmentation for higher accuracy
- A `--model` / `--epochs` command-line interface alongside the menu
- A confidence threshold that flags uncertain predictions
- Saving prediction results to a CSV file
- Unit tests for the preprocessing pipeline

## License

No license has been added to this repository yet — all rights are reserved by the author by
default. MNIST is used under its own terms (LeCun, Cortes, Burges).
