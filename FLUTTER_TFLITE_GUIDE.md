# Flutter TFLite Integration Guide — OptiTrace Eye Disease Classifier

**Purpose:** This document explains how to match Python TFLite predictions in Flutter. Prediction mismatches almost always come from preprocessing differences.

---

## 1. Model Specifications (Verified)

| Property | Value |
|----------|-------|
| **Input shape** | `[1, 224, 224, 3]` (batch, height, width, channels) |
| **Input dtype** | `float32` |
| **Input value range** | **0.0 to 255.0** (raw pixel values, not normalized) |
| **Output shape** | `[1, 4]` → 4 class probabilities |
| **Output dtype** | `float32` |
| **Layout** | NHWC (channels last) |

---

## 2. Critical: Do NOT Normalize Input

**The model expects raw pixel values in [0, 255].**

MobileNetV3’s preprocessing (`Rescaling(scale=1/127.5, offset=-1)`) is already inside the model. The Python code uses `preprocess_input` from MobileNetV3, which is a pass-through and does nothing.

**Wrong (common mistake):**
```dart
// ❌ WRONG - normalizing to [-1, 1] or [0, 1]
pixelValue = (pixelValue / 127.5) - 1;   // [-1, 1] - WRONG
pixelValue = pixelValue / 255.0;         // [0, 1] - WRONG
```

**Correct:**
```dart
// ✅ CORRECT - keep raw 0-255 values as float32
pixelValue = pixelValue.toDouble();      // 0.0 to 255.0
```

---

## 3. Preprocessing Steps (Must Match Python)

The Python pipeline does:

1. Load image (e.g. JPEG) → RGB
2. Resize to **224×224** (bilinear or similar)
3. Convert to float32, shape `(1, 224, 224, 3)`
4. Keep values in **0–255** (no normalization)

Flutter equivalent:

1. Decode image as RGB
2. Resize to 224×224
3. Extract pixel values as `Float32List` in NHWC order
4. Values stay 0.0–255.0

---

## 4. NHWC Layout

Layout must be **channels last**:

```
[batch][row][col][channel]
[0][y][x][0] = R
[0][y][x][1] = G  
[0][y][x][2] = B
```

Index order: `(batchIndex * 224 * 224 * 3) + (y * 224 * 3) + (x * 3) + channel`

---

## 5. Color: RGB, Not BGR

Use RGB channel order. If using `image` or camera packages, ensure no BGR conversion.

---

## 6. Label Order (Class Indices)

| Index | Label           |
|-------|-----------------|
| 0     | Cataract        |
| 1     | Conjunctivitis  |
| 2     | Normal          |
| 3     | Uveitis         |

`argmax` of the output array gives the predicted class index.

---

## 7. Example Flutter Preprocessing (Pseudocode)

```dart
import 'dart:typed_data';

/// Build input buffer: shape [1, 224, 224, 3], float32, values 0-255
Float32List preprocessImage(Image image) {
  const int size = 224;
  final input = Float32List(1 * size * size * 3);
  int idx = 0;
  
  // Resize image to 224x224 first (use image package or similar)
  final resized = copyResize(image, width: size, height: size);
  
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      final pixel = resized.getPixel(x, y);
      // RGB order, raw 0-255 as float32
      input[idx++] = (pixel.r.toInt() & 0xFF).toDouble();
      input[idx++] = (pixel.g.toInt() & 0xFF).toDouble();
      input[idx++] = (pixel.b.toInt() & 0xFF).toDouble();
    }
  }
  return input;
}
```

---

## 8. Checklist for Debugging Mismatches

| Check | Expected |
|-------|----------|
| Input shape | `[1, 224, 224, 3]` |
| Input dtype | `float32` |
| Input range | 0.0–255.0 (no normalization) |
| Layout | NHWC |
| Color order | RGB |
| Label indices | 0=Cataract, 1=Conjunctivitis, 2=Normal, 3=Uveitis |
| Resize method | Bilinear or similar (small differences may be acceptable) |

---

## 9. Test Image Reference

Use an image from `demo/` for a comparison test. Run `python predict_image.py` to get the Python reference output.

If Flutter gives different probabilities or a different argmax, the cause is likely preprocessing.

---

## 10. Common Mismatch Causes

1. **Normalizing to [0, 1] or [-1, 1]** — most common; do not normalize.
2. **Wrong channel order** — BGR instead of RGB.
3. **Wrong layout** — NCHW instead of NHWC.
4. **Wrong resize** — different size or interpolation.
5. **Integer vs float** — input must be `float32`, not `uint8`/`int32`.
6. **Label order** — ensure indices match the table above.

---

## 11. Python Reference (for comparison)

```python
# test_tflite.py - what Python does
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)           # RGB, 0-255, shape (224,224,3)
x = np.expand_dims(x, axis=0)         # (1, 224, 224, 3)
x = x.astype("float32")               # float32
x = preprocess_input(x)               # pass-through, no change
# Result: float32, [0,255], NHWC
```
