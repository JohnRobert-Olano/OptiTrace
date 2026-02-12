
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np

# Create dummy data: Black (0), Middle (127), White (255)
data = np.array([0, 127, 255], dtype=np.float32)
data = np.expand_dims(data, axis=-1)
data = np.expand_dims(data, axis=0) # (1, 3, 1)

# Preprocess
out = preprocess_input(data.copy())

print("Input: [0, 127, 255]")
print("Output:", out.flatten())

# Check scale
print("Min:", out.min())
print("Max:", out.max())
