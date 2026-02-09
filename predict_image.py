"""
OptiTrace - Image prediction using TFLite model.
Uses model.tflite for inference (avoids Keras .keras load issues across versions).
"""
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = (224, 224)

# Load labels
def _load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_and_prepare(img_path: str):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype("float32")
    x = preprocess_input(x)
    return x


def predict_single(img_path: str):
    print(f"\nImage: {img_path}")
    if not os.path.exists(img_path):
        print("  -> File not found.")
        return

    if not os.path.exists(TFLITE_PATH):
        print(f"  -> TFLite model not found: {TFLITE_PATH}")
        print("     Run 'python convert_to_tflite.py' first.")
        return

    try:
        labels = _load_labels()
        x = load_and_prepare(img_path)

        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details["index"])[0]

        top_idx = int(np.argmax(probs))
        top_class = labels[top_idx]
        top_prob = float(probs[top_idx])

        print(f"Predicted class: {top_class} (confidence: {top_prob:.3f})")
        print("All class probabilities:")
        for name, p in zip(labels, probs):
            print(f"  {name:15s}: {p:.3f}")

    except Exception as e:
        print(f"  -> Prediction failed: {e}")

if __name__ == "__main__":
    demo_dir = os.path.join(BASE_DIR, "demo")
    
    # Update this list with your actual file names
    images = [
        os.path.join(demo_dir, "normal_example.jpg"),
        os.path.join(demo_dir, "disease_example.jpg"),
        os.path.join(demo_dir, "blurry_example.jpg"),
    ]
    
    for p in images:
        predict_single(p)