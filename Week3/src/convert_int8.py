# Week3/src/convert_int8.py
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob

keras_path = "Week2/outputs/best_model_week2.h5"
tflite_out = "Week3/outputs/model_int8.tflite"
rep_img_dir = "Week3/sample_images"  # add a few images here

def representative_dataset_gen():
    files = glob.glob(os.path.join(rep_img_dir, "*.*"))
    for i, f in enumerate(files):
        if i >= 100: break
        img = Image.open(f).convert("RGB").resize((224,224))  # use the same input size you trained with
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        yield [arr]

if not os.path.exists(keras_path):
    raise FileNotFoundError(keras_path)

model = tf.keras.models.load_model(keras_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# enforce integer-only (for full integer quant)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8   # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

tflite_model = converter.convert()
os.makedirs(os.path.dirname(tflite_out), exist_ok=True)
with open(tflite_out, "wb") as f:
    f.write(tflite_model)

size_bytes = os.path.getsize(tflite_out)
print(f"Saved INT8 TFLite model to: {tflite_out} (size {size_bytes/1e6:.2f} MB)")
