"""
Compare predictions from Keras (.keras) and TFLite (model.tflite) on the same images.
Use this to verify both models produce consistent results.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Which Keras model to compare against TFLite:
# - "best_model.keras"        = Stage A (head only, before fine-tuning)
# - "best_model_finetuned.keras" = Best checkpoint during fine-tuning
# - "final_model.keras"       = Final saved model (same source as model.tflite)
KERAS_MODEL = "best_model.keras"
KERAS_PATH = os.path.join(BASE_DIR, "models", KERAS_MODEL)
TFLITE_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = (224, 224)


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_and_prepare(img_path: str):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype("float32")
    x = preprocess_input(x)
    return x


def predict_keras(model, x):
    return model.predict(x, verbose=0)[0]


def predict_tflite(interpreter, x):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


def main():
    demo_dir = os.path.join(BASE_DIR, "demo")
    images = [
        os.path.join(demo_dir, "normal_example.jpg"),
        os.path.join(demo_dir, "disease_example.jpg"),
        os.path.join(demo_dir, "blurry_example.jpg"),
    ]

    # Filter to existing images
    images = [p for p in images if os.path.exists(p)]
    if not images:
        print("No demo images found. Add images to demo/ folder.")
        return

    labels = load_labels()

    # Load Keras model (optional - may fail with version mismatch)
    keras_model = None
    if os.path.exists(KERAS_PATH):
        try:
            keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)
            print(f"Loaded Keras model: {KERAS_PATH}\n")
        except Exception as e:
            print(f"Could not load Keras model: {e}")
            print("(Skipping Keras comparison - TFLite only)\n")
    else:
        print(f"Keras model not found: {KERAS_PATH}\n")

    # Load TFLite
    if not os.path.exists(TFLITE_PATH):
        print(f"TFLite model not found: {TFLITE_PATH}")
        print("Run 'python convert_to_tflite.py' first.")
        return

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    print(f"Loaded TFLite model: {TFLITE_PATH}\n")
    print("=" * 70)

    for img_path in images:
        print(f"\nImage: {os.path.basename(img_path)}")
        x = load_and_prepare(img_path)

        # TFLite prediction
        tflite_probs = predict_tflite(interpreter, x)
        tflite_idx = int(np.argmax(tflite_probs))
        tflite_label = labels[tflite_idx]
        tflite_conf = float(tflite_probs[tflite_idx])

        print(f"\n  TFLite:  {tflite_label:15s} ({tflite_conf:.4f})")
        print(f"           Probs: [{', '.join(f'{p:.3f}' for p in tflite_probs)}]")

        if keras_model is not None:
            keras_probs = predict_keras(keras_model, x)
            keras_idx = int(np.argmax(keras_probs))
            keras_label = labels[keras_idx]
            keras_conf = float(keras_probs[keras_idx])

            print(f"\n  Keras:   {keras_label:15s} ({keras_conf:.4f})")
            print(f"           Probs: [{', '.join(f'{p:.3f}' for p in keras_probs)}]")

            # Compare
            match = tflite_idx == keras_idx
            max_diff = float(np.max(np.abs(tflite_probs - keras_probs)))
            print(f"\n  Match:   {match}  |  Max prob diff: {max_diff:.4f}")

        print("-" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()
