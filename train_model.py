"""
OptiTrace - MobileNet Transfer Learning for External Eye Disease Classification
Undergraduate Thesis: "MobileNet Transfer Learning Approach in Classifying Common External Eye Diseases"

Classes: Cataract, Conjunctivitis, Normal (Healthy Eye), Uveitis
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (adjust BASE_DIR if running from a different location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Image and model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["cataract", "conjunctivitis", "normal", "uveitis"]

# Training parameters
EPOCHS_HEAD = 30
EPOCHS_FINE_TUNE = 20
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE_TUNE = 1e-5

# Fine-tuning: unfreeze the last N layers (BatchNorm stays frozen)
FINE_TUNE_LAST_N_LAYERS = 30


# =============================================================================
# DATA GENERATORS
# =============================================================================

def get_train_datagen():
    """Training data generator with augmentation."""
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


def get_val_datagen():
    """Validation data generator (no augmentation)."""
    return ImageDataGenerator(preprocessing_function=preprocess_input)


def create_data_generators():
    """Create train, validation, and (optional) test data generators."""
    train_datagen = get_train_datagen()
    val_datagen = get_val_datagen()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42,
    )

    test_generator = None
    if os.path.exists(TEST_DIR):
        test_generator = val_datagen.flow_from_directory(
            TEST_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=CLASS_NAMES,
            shuffle=False,
            seed=42,
        )

    return train_generator, val_generator, test_generator


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_model():
    """
    Build MobileNetV3-Large transfer learning model.
    - Base: MobileNetV3Large (ImageNet pre-trained), frozen
    - Head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(4, softmax)
    """
    # Load pre-trained MobileNetV3-Large (exclude top classification layer)
    base_model = MobileNetV3Large(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    # Freeze all base model layers
    base_model.trainable = False

    # Build custom classification head
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="OptiTrace_EyeDisease_Classifier")

    return model, base_model


def compute_class_weights(train_generator):
    """
    Compute balanced class weights from the training generator labels.
    Helps when you have class imbalance.
    """
    # train_generator.classes is an array of integer class indices (one per image)
    y = train_generator.classes
    class_counts = {i: int((y == i).sum()) for i in range(NUM_CLASSES)}
    total = int(len(y))
    weights = {}
    for i in range(NUM_CLASSES):
        # Balanced weight: total / (num_classes * count_i)
        count_i = max(class_counts[i], 1)
        weights[i] = total / (NUM_CLASSES * count_i)

    print("\n    Train class counts:")
    inv_map = {v: k for k, v in train_generator.class_indices.items()}
    for i in range(NUM_CLASSES):
        print(f"      - {inv_map[i]}: {class_counts[i]}")
    print("    Computed class weights:", weights)

    return weights


def set_fine_tune(base_model, last_n_layers):
    """
    Unfreeze the last N layers for fine-tuning.
    Keep BatchNormalization layers frozen for stability.
    """
    base_model.trainable = True
    if last_n_layers <= 0:
        # If user sets 0, keep base frozen
        base_model.trainable = False
        return

    # Freeze all layers except last N
    for layer in base_model.layers[:-last_n_layers]:
        layer.trainable = False

    # BatchNorm layers: keep frozen (common best practice)
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def evaluate_split(model, generator, class_names, split_name):
    """Print and save confusion matrix + classification report for a generator split."""
    generator.reset()
    probs = model.predict(generator, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[4] {split_name} classification report:\n")
    print(report)
    print(f"[4] {split_name} confusion matrix (rows=true, cols=pred):\n")
    print(cm)

    # Save to disk for thesis documentation
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    safe_name = split_name.lower().replace(" ", "_")
    report_path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}_classification_report.txt")
    cm_path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}_confusion_matrix.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm))
        f.write("\n")
    print(f"\n    Saved report to: {report_path}")
    print(f"    Saved confusion matrix to: {cm_path}")


# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("OptiTrace - Eye Disease Classification Training")
    print("=" * 60)

    # Verify data directories exist
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            "Please create the folder structure and add your images."
        )
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_DIR}\n"
            "Please create the folder structure and add your images."
        )

    # Create data generators
    print("\n[1] Loading data generators...")
    train_generator, val_generator, test_generator = create_data_generators()
    print(f"    Training samples: {train_generator.samples}")
    print(f"    Validation samples: {val_generator.samples}")
    print(f"    Classes: {list(train_generator.class_indices.keys())}")
    if test_generator is not None:
        print(f"    Test samples: {test_generator.samples}")

    # Build model
    print("\n[2] Building model (MobileNetV3-Large + custom head)...")
    model, base_model = build_model()

    # Compile model (head training)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks_head = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    class_weights = compute_class_weights(train_generator)

    # Stage A: Train classifier head
    print("\n[3A] Training classifier head (base frozen)...")
    history_head = model.fit(
        train_generator,
        epochs=EPOCHS_HEAD,
        validation_data=val_generator,
        callbacks=callbacks_head,
        class_weight=class_weights,
        verbose=1,
    )

    # Stage B: Fine-tuning
    print("\n[3B] Fine-tuning (unfreezing top layers)...")
    set_fine_tune(base_model, FINE_TUNE_LAST_N_LAYERS)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_finetune = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "best_model_finetuned.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history_finetune = model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=val_generator,
        callbacks=callbacks_finetune,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate with detailed metrics
    evaluate_split(model, val_generator, CLASS_NAMES, "Validation")
    if test_generator is not None and test_generator.samples > 0:
        evaluate_split(model, test_generator, CLASS_NAMES, "Test")
    else:
        print("\n[4] Test evaluation skipped (no 'data/test' images found).")

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "final_model.keras")
    model.save(final_path)
    print(f"\n[5] Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
