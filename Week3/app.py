import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import json

st.title("WasteWise — Demo (Week 3)")

uploaded = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
if uploaded:
    st.image(uploaded, caption='Uploaded Image', use_column_width=True)

    temp_path = "Week3/temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    model_path = "Week2/outputs/best_model_week2.h5"
    if not os.path.exists(model_path):
        st.error("Model not found: " + model_path)
    else:
        model = load_model(model_path)
        img = image.load_img(temp_path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr)
        idx = int(preds.argmax())

        mapping_path = "Week2/outputs/class_indices.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                class_map = json.load(f)
            inv_map = {v: k for k, v in class_map.items()}
            label = inv_map.get(idx, str(idx))
        else:
            label = str(idx)

        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: {float(preds.max()):.3f}")
