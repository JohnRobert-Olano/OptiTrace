import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFLITE_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = (224, 224)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype("float32")
    x = preprocess_input(x)  # SAME preprocessing as training
    return x


def predict_with_tflite(img_path):
    print(f"\nImage: {img_path}")
    if not os.path.exists(img_path):
        print("  -> File not found.")
        return

    labels = load_labels(LABELS_PATH)
    x = load_and_preprocess(img_path)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Show I/O info once
    print("Input details:", input_details[0])
    print("Output details:", output_details[0])

    # Set input
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    # Get output
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_prob = float(probs[top_idx])

    print(f"Predicted class: {top_label} (confidence: {top_prob:.3f})")
    print("All class probabilities:")
    for label, p in zip(labels, probs):
        print(f"  {label:15s}: {p:.3f}")


if __name__ == "__main__":
    # Change these to your real test images
    demo_dir = os.path.join(BASE_DIR, "demo")
    images = [
        os.path.join(demo_dir, "normal_example.jpg"),
        os.path.join(demo_dir, "disease_example.jpg"),
        os.path.join(demo_dir, "blurry_example.jpg"),
    ]
    for p in images:
        predict_with_tflite(p)