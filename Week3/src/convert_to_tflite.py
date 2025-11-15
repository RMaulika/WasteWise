# Week3/src/convert_to_tflite.py
import tensorflow as tf
import os

# change this path if your .h5 has a different name
keras_path = "Week2/outputs/best_model_week2.h5"
tflite_out = "Week3/outputs/model_opt.tflite"

if not os.path.exists(keras_path):
    raise FileNotFoundError(f"keras model not found: {keras_path}")

print("Loading Keras model from:", keras_path)
model = tf.keras.models.load_model(keras_path)
print("Converting to TFLite (post-training quantization, default optimizations)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# enable default optimizations (size & latency)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# If you want float16 quantization (smaller) uncomment:
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

os.makedirs(os.path.dirname(tflite_out), exist_ok=True)
with open(tflite_out, "wb") as f:
    f.write(tflite_model)

size_bytes = os.path.getsize(tflite_out)
print(f"Saved TFLite model to: {tflite_out}  (size: {size_bytes} bytes = {size_bytes/1e6:.2f} MB)")
