# OptiTrace: Python AI Developer Standards 

## Core Architecture
- **Language:** Python 3.10+.
- **Typing:** Strict type hinting is required for all functions and class methods using the `typing` module. Do not generate untyped Python code.
- **Formatting:** Adhere strictly to PEP 8 standards. Use `Black` formatting rules natively.

## Machine Learning Integration (MobileNetV3)
- **Memory Management:** Ensure tensors and model weights are properly loaded and cleared from memory when no longer in use to prevent memory leaks during batch processing.
- **Error Handling:** Wrap model inference calls in `try/except` blocks to gracefully handle corrupted image data or dimension mismatches.
- **Modularity:** Separate the model loading logic from the inference/prediction logic.

## Workflow & Code Generation
- Do not write monolithic scripts. Break logic into distinct modules (e.g., `model_loader.py`, `inference.py`, `utils.py`).
- Always document functions with Google-style docstrings explaining args, returns, and exceptions raised.
