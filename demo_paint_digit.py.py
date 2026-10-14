"""
Live demo: draw a digit in Paint, save it, and let our SVM read it.

Usage:  python demo_paint_digit.py my_digit.png
        python demo_paint_digit.py            (opens a file dialog)

Drawing tips (Paint): white canvas, black pen/brush, THICK stroke (size 8 to 12),
one big digit in the middle, save as PNG.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

PAD = 0.05   # empty border around the digit, as a fraction of its size


def train_model():
    """Same final model as the notebook (RBF, C=1, gamma=0.1) + probability estimates."""
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([("scaler", MinMaxScaler()),
                      ("svm", SVC(kernel="rbf", C=1, gamma=0.1, probability=True, random_state=42))])
    model.fit(X_tr, y_tr)
    return model, model.score(X_te, y_te)


def load_gray(path):
    """Open any image (PNG/JPG/BMP) as a grayscale array, transparent areas treated as white."""
    img = Image.open('test_sample5.png')
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.array(img.convert("L"), dtype=float)


def to_8x8(gray):
    """Turn a big drawing into the 8x8, 0-to-16 format the model was trained on."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.median(border) > 127:          # white paper, dark ink -> flip so ink is bright
        gray = 255 - gray
    gray[gray < 40] = 0                  # remove faint noise
    ys, xs = np.where(gray > 0)
    if len(ys) == 0:
        raise ValueError("No digit found. Draw with a dark pen on a white background.")
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]      # crop to the digit
    h, w = gray.shape
    side = int(max(h, w) * (1 + 2 * PAD))                            # square canvas, digit centred
    canvas = np.zeros((side, side))
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = gray
    small = Image.fromarray(canvas.astype(np.uint8)).resize((8, 8), Image.BOX)   # area average
    small = np.array(small, dtype=float)
    return small / small.max() * 16                                  # same 0-16 scale as the data


def predict_drawing(model, path, show=True):
    gray = load_gray(path)
    x8 = to_8x8(gray.copy())
    proba = model.predict_proba(x8.reshape(1, -1))[0]
    digit = int(np.argmax(proba))
    print(f"\nThe model reads this as:  {digit}   (confidence {100 * proba[digit]:.1f}%)")
    print("Top 3:", ", ".join(f"{d}: {100 * proba[d]:.1f}%" for d in np.argsort(-proba)[:3]))
    if show:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1, 2]})
        ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Your drawing"); ax[0].axis("off")
        ax[1].imshow(x8, cmap="gray_r"); ax[1].set_title("What the model sees (8x8)"); ax[1].axis("off")
        colors = ["tab:green" if d == digit else "lightgray" for d in range(10)]
        bars = ax[2].bar(range(10), 100 * proba, color=colors)
        ax[2].bar_label(bars, fmt="%.0f", fontsize=8)
        ax[2].set_xticks(range(10)); ax[2].set_ylim(0, 110)
        ax[2].set_xlabel("Digit"); ax[2].set_ylabel("Confidence (%)")
        ax[2].set_title(f"Prediction: {digit}  ({100 * proba[digit]:.1f}% confident)")
        plt.tight_layout(); plt.show()
    return digit, proba


if __name__ == "__main__":
    print("Training the SVM (takes a second)...")
    model, test_acc = train_model()
    print(f"Ready. Accuracy on our 360 held-out test images: {100 * test_acc:.1f}%")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import tkinter as tk
        from tkinter import filedialog
        tk.Tk().withdraw()
        path = filedialog.askopenfilename(title="Choose your digit image")
    predict_drawing(model, path)