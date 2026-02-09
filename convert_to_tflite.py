import os
import pathlib
import tensorflow as tf

# Filenames
MODEL_KERAS = "best_model_finetuned.keras"
TFLITE_MODEL = "model.tflite"
LABELS_FILE = "labels.txt"

# Class labels in alphabetical order (must match train_model.py CLASS_NAMES order)
CLASS_LABELS = [
    "Cataract",
    "Conjunctivitis",
    "Normal",
    "Uveitis",
]


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main():
    base_dir = pathlib.Path(__file__).resolve().parent
    # Model saved by train_model.py goes to models/
    keras_path = base_dir / "models" / MODEL_KERAS
    tflite_path = base_dir / TFLITE_MODEL
    labels_path = base_dir / LABELS_FILE

    # 1. Load the trained Keras model
    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")
    print(f"Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    # 2–3. Convert to TFLite with default optimizations (quantization)
    print("Converting to TensorFlow Lite with default optimizations...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # default optimizations (quantization)
    tflite_model = converter.convert()

    # 4. Save as model.tflite
    tflite_path.write_bytes(tflite_model)
    print(f"Saved TFLite model to: {tflite_path}")

    # 5. Generate labels.txt in alphabetical order
    print(f"Writing labels to: {labels_path}")
    with labels_path.open("w", encoding="utf-8") as f:
        for label in CLASS_LABELS:
            f.write(label + "\n")

    # 6. Print input/output file sizes
    keras_size = os.path.getsize(keras_path)
    tflite_size = os.path.getsize(tflite_path)

    print("\n=== Model Size Report ===")
    print(f"Keras model ({MODEL_KERAS}): {keras_size} bytes ({bytes_to_mb(keras_size):.2f} MB)")
    print(f"TFLite model ({TFLITE_MODEL}): {tflite_size} bytes ({bytes_to_mb(tflite_size):.2f} MB)")
    if keras_size > 0:
        compression = 100.0 * (1.0 - tflite_size / keras_size)
        print(f"Size reduction: {compression:.2f}%")
    print("=========================\n")


if __name__ == "__main__":
    main()