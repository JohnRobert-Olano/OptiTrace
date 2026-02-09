# OptiTrace - Project Folder Structure

## Directory Layout for Eye Disease Classification

Place your dataset images in the following structure. Keras `flow_from_directory` expects subdirectories named after each class.

```
OptiTrace/
│
├── data/
│   ├── train/                    # Training images (~70-80% of dataset)
│   │   ├── cataract/             # Cataract eye images
│   │   ├── conjunctivitis/       # Conjunctivitis (pink eye) images
│   │   ├── normal/               # Healthy/normal eye images
│   │   └── uveitis/              # Uveitis eye images
│   │
│   └── val/                      # Validation images (~20-30% of dataset)
│       ├── cataract/
│       ├── conjunctivitis/
│       ├── uveitis/
│       └── normal/
│
│   └── test/                     # Test images (final thesis evaluation)
│       ├── cataract/
│       ├── conjunctivitis/
│       ├── uveitis/
│       └── normal/
│
├── models/                       # Saved model checkpoints (created during training)
├── logs/                         # TensorBoard logs (optional)
├── train_model.py                # Main training script
├── requirements.txt              # Python dependencies
└── PROJECT_STRUCTURE.md          # This file
```

## Dataset Guidelines

1. **Folder names must match exactly** (lowercase, underscores): `cataract`, `conjunctivitis`, `normal`, `uveitis`
2. **Supported formats**: JPG, JPEG, PNG, BMP
3. **Recommended split**: 70% train, 15% validation, 15% test (stratified by class if possible)
4. **Minimum images per class**: Aim for at least 50-100 images per class for reasonable transfer learning results

## Creating the Folders

Run this in PowerShell from the project root:

```powershell
$classes = @("cataract", "conjunctivitis", "normal", "uveitis")
foreach ($split in @("train", "val", "test")) {
    foreach ($cls in $classes) {
        New-Item -ItemType Directory -Force -Path "data\$split\$cls"
    }
}
```
