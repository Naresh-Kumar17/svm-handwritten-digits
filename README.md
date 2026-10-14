# SVM on Handwritten Digits

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange) ![Test accuracy](https://img.shields.io/badge/test%20accuracy-99.2%25-brightgreen) ![Course](https://img.shields.io/badge/DS2001-Assignment%2002-lightgrey)

Can a machine read your handwriting? We use **Support Vector Machines** to recognize handwritten digits, and explain the math, build our own SVM from scratch, tune it, break it with noise, and test it on a digit **you draw yourself**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/svm-handwritten-digits/blob/main/svm_digits_M0_M11.ipynb)

**Jump to:** [Quick start](#-quick-start) | [Try the live demo](#-try-it-draw-a-digit) | [Results](#-results-at-a-glance) | [Notebook map](#-notebook-map) | [Findings](#-key-findings) | [Limitations](#-limitations)

> DS2001 Introduction to Data Science, Fall 2026, Assignment 02 (group project)

---

## The idea in 30 seconds

1. An 8x8 image of a digit becomes **64 numbers** (one per pixel).
2. SVM finds the boundary with the **widest gap (margin)** between classes.
3. A **kernel** lets that boundary curve, without ever computing the extra features.
4. We test it on 360 images the model never saw: **99.2% accuracy**.

---

## ⚡ Quick start

```bash
git clone https://github.com/<your-username>/svm-handwritten-digits.git
cd svm-handwritten-digits
pip install -r requirements.txt
jupyter notebook svm_digits_M0_M11.ipynb
```

Choose **Restart and Run All**. It finishes in about a minute. (The interactive 3D plots need `plotly`; without it the notebook prints an install hint and the static plots still work.)

## ✏️ Try it: draw a digit

1. In Paint: white canvas, black pen, **thick stroke** (size 8 to 12).
2. Draw one big digit in the middle and save it as PNG.
3. Run:

```bash
python demo_paint_digit.py my_digit.png
```

You will see your drawing, what the model sees (8x8), and its confidence for every digit.

<details>
<summary><b>Why was my digit misread?</b></summary>

- The percentage shown is the model's **confidence** for that one drawing, not its accuracy.
- It learned from 8x8 images by a limited set of writers, so unusual styles can fool it.
- A "1" drawn with a base or flag is the hardest to read. Try a plain stroke.
- Make the digit large, centered and thick, and use a plain white background.
</details>

---

## 📊 Results at a glance

Random seed 42, stratified 80/20 split (1,437 train, 360 test images).

| Result | Value |
|---|---|
| Final model | RBF SVM, C = 1, gamma = 0.1 |
| Test accuracy | **99.2%** (3 errors in 360), macro F1 0.992 |
| 5-fold CV accuracy | 98.8% |
| Linear SVM (CV) | 97.8% |
| Logistic Regression (test) | 95.8% |
| k-NN, k = 3 (test) | 98.6% |
| Our scratch SVM, digits 3 vs 8 | 100% test accuracy |
| Our scratch one-vs-rest, 10 digits | 95.8% test accuracy |
| Circles dataset, linear vs RBF | 60% vs 100% |
| Support vectors in final model | 664 of 1,437 images (46%) |

---

## 🗺️ Notebook map

<details>
<summary><b>Click to see all 12 modules</b></summary>

| Module | What you will see |
|---|---|
| M0 | Data exploration: sample digits, class balance, average image per digit |
| M1 | Preprocessing: stratified split, scaling inside a Pipeline |
| M2 | Margin intuition on Iris (2D), effect of C |
| M3 | Math notes and a worked 4-point example checked against sklearn |
| M4 | Linear SVM from scratch (NumPy), digits 3 vs 8 |
| M5 | One-vs-rest wrapper over all 10 digits |
| M6 | Kernel trick on moons and circles, with a 3D lift |
| M7 | Linear, polynomial and RBF kernels on digits |
| M8 | Tuning C and gamma with GridSearchCV and 5-fold CV |
| M9 | Confusion matrices, per-class metrics, baselines |
| M10 | Support vectors as images, misclassified digits, PCA in 2D and 3D |
| M11 | Results summary, noise experiment, findings, limitations |
</details>

<details>
<summary><b>Datasets used</b> (all built into scikit-learn, nothing to download)</summary>

| Dataset | Used for |
|---|---|
| Digits (`load_digits`): 1,797 images, 64 features, 10 classes | Main demo and final results |
| Iris (petal length and width only) | Margin, support vectors, effect of C |
| `make_moons` (synthetic) | Linear vs RBF boundary, effect of gamma |
| `make_circles` (synthetic) | Kernel trick and the 3D lift z = x² + y² |
</details>

<details>
<summary><b>Repository contents</b></summary>

| File | Description |
|---|---|
| `svm_digits_M0_M11.ipynb` | Main notebook: implementation, plots, analysis |
| `SVM_Digits_Presentation.pptx` | 20-slide class presentation |
| `demo_paint_digit.py` | Live demo: draw a digit and let the model read it |
| `requirements.txt` | Python packages needed |
</details>

---

## 🔍 Key findings

- **Scaling did not help on digits.** No scaling and MinMax both gave 99.2%; StandardScaler gave 97.5%. All pixels already share one scale.
- **Gamma is the sensitive setting**, and C matters much less. The default gamma was already near the best value.
- **Kernels beat a straight boundary.** Linear was clearly worst in cross-validation; polynomial and RBF are within noise of each other.
- **SVM clearly beats Logistic Regression** but only edges k-NN, within noise.
- **The tuned model breaks on noise it never saw.** Accuracy fell to 78.9% with mild noise; training on noisy copies kept it above 91% even with strong noise.
- The most confused digit pairs (cross-validated) were 1/8, 5/9, 8/9 and 3/5, but the counts are small.

<details>
<summary><b>Self-check: can you answer these after reading the notebook?</b></summary>

- [ ] What are support vectors, and why can you delete the other points?
- [ ] What do C and gamma each control?
- [ ] Why does the kernel trick never compute the extra features?
- [ ] Why did StandardScaler hurt on pixel data?
- [ ] Why is the noise result a limitation, and what fixed it?
</details>

---

## ⚠️ Limitations

- Sensitive to the choice of C and gamma, and fragile to noise it was not trained on.
- No probabilities by default (the demo uses an extra calibration step).
- Treats pixels as a flat list and ignores the 2D layout, so convolutional networks are the usual choice for large image datasets.
- The test set has only 360 images, so one image equals about 0.3 percentage points.

## Tech stack

Python, NumPy, pandas, Matplotlib, scikit-learn, Plotly, Pillow.
