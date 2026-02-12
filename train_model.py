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

import numpy as np
import cv2

class ClipperSequence(keras.utils.Sequence):
    """
    Custom Data Generator that applies "Clipper" preprocessing:
    Clean_Image = Raw_Image * (Mask / 255)
    """
    def __init__(self, image_dir, mask_dir, batch_size, img_size, 
                 shuffle=True, augment=False, class_names=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Discover classes (subdirectories)
        if class_names is None:
            self.classes = sorted([d for d in os.listdir(image_dir) 
                                 if os.path.isdir(os.path.join(image_dir, d))])
        else:
            self.classes = class_names
            
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            cls_image_dir = os.path.join(image_dir, cls)
            if not os.path.exists(cls_image_dir):
                continue
                
            for fname in os.listdir(cls_image_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(cls_image_dir, fname))
                    self.labels.append(self.class_indices[cls])
        
        self.samples = len(self.image_paths)
        self.indices = np.arange(self.samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 1. Load Image
            # OpenCV loads as BGR, convert to RGB
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                # Create black placeholder or skip (skipping breaks batch size, so placeholder)
                img = np.zeros((*self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
            
            # 2. Load Mask
            # Construct mask path: replace 'data' dir with 'data_masks' logic
            # We assume structure: .../class/filename.ext
            parent_dir, fname = os.path.split(img_path)
            _, class_name = os.path.split(parent_dir)
            
            mask_fname = os.path.splitext(fname)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, class_name, mask_fname)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                # If mask missing, assume fully visible (all ones) or clean (all zeros)?
                # Usually better to warn. For now, assume full visibility -> all 255
                # But requirement says "match 1-to-1". Let's assume they exist.
                # Fallback: full white mask
                mask = np.ones(self.img_size, dtype=np.uint8) * 255
            else:
                mask = cv2.resize(mask, self.img_size)
            
            # 3. Apply Clipper: Image * (Mask / 255)
            # Ensure types match for multiplication
            img = img.astype(np.float32)
            mask = mask.astype(np.float32) / 255.0
            
            # Expand mask dimensions to match image channels
            mask = np.expand_dims(mask, axis=-1)
            
            clipped_img = img * mask
            
            # 4. Augmentation (Simple version)
            # Only if augment=True (training)
            if self.augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    clipped_img = np.fliplr(clipped_img)
                
                # Random brightness (simple scaling)
                brightness_factor = np.random.uniform(0.8, 1.2)
                clipped_img = clipped_img * brightness_factor
                clipped_img = np.clip(clipped_img, 0, 255)
            
            # 5. Preprocess for MobileNetV3
            # Expected input is usually -1 to 1 or similar depending on function
            # preprocess_input is imported from mobilenet_v3
            processed_img = preprocess_input(clipped_img.copy())
            
            batch_images.append(processed_img)
            
            # One-hot encode label
            one_hot = np.zeros(NUM_CLASSES)
            one_hot[label] = 1
            batch_labels.append(one_hot)
            
        return np.array(batch_images), np.array(batch_labels)

def save_debug_examples(generator, output_dir="debug", num_examples=5):
    """
    Save examples of clipped images to verify the preprocessing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n[Verification] Saving {num_examples} processed debug images to '{output_dir}'...")
    
    # Get one batch
    images, labels = generator[0]
    
    # Invert preprocessing for visualization
    # MobileNetV3 preprocess usually maps [0, 255] -> [-1, 1] or similar
    # But tf.keras.applications.mobilenet_v3.preprocess_input documentation says:
    # "The inputs will be scaled between -1 and 1" or sometimes 0-1 depending on mode.
    # It acts on float inputs.
    # Let's check common behavior: (x - 127.5) / 127.5?
    # To restore roughly: (x + 1) * 127.5
    
    for i in range(min(num_examples, len(images))):
        img_processed = images[i]
        
        # MobileNetV3 preprocessing in this version/setup seems to keep range [0, 255]
        # or [0, 1]. Based on check_preprocess.py, it was [0, 255].
        # So we just clip and cast.
        img_display = np.clip(img_processed, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV saving
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        label_idx = np.argmax(labels[i])
        class_name = CLASS_NAMES[label_idx]
        
        out_path = os.path.join(output_dir, f"clipper_example_{i}_{class_name}.png")
        cv2.imwrite(out_path, img_display)
        print(f"  - Saved: {out_path}")


def create_data_generators():
    """Create train, validation, and (optional) test data generators using ClipperSequence."""
    
    # Define Mask Directories
    # Assuming parallel structure:
    # data/train -> data_masks/train
    TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data_masks", "train")
    VAL_MASK_DIR = os.path.join(BASE_DIR, "data_masks", "val")
    TEST_MASK_DIR = os.path.join(BASE_DIR, "data_masks", "test")
    
    train_generator = ClipperSequence(
        image_dir=TRAIN_DIR,
        mask_dir=TRAIN_MASK_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=True,
        augment=True,
        class_names=CLASS_NAMES
    )
    
    val_generator = ClipperSequence(
        image_dir=VAL_DIR,
        mask_dir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=False,
        augment=False,
        class_names=CLASS_NAMES
    )
    
    test_generator = None
    if os.path.exists(TEST_DIR):
        test_generator = ClipperSequence(
            image_dir=TEST_DIR,
            mask_dir=TEST_MASK_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            shuffle=False,
            augment=False,
            class_names=CLASS_NAMES
        )
            
    # Run verification once
    save_debug_examples(train_generator)
            
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
    # Custom Sequence stores labels in self.labels
    y = np.array(train_generator.labels)
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
    # generator.reset() # Not needed for Sequence as it's handled by index access
    probs = model.predict(generator, verbose=0)
    y_pred = probs.argmax(axis=1)
    # Custom Sequence: labels are in .labels
    y_true = generator.labels

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
