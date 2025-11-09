# Week2/src/infer_finetuned.py
import sys, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_image(model_path, img_path, class_indices_path=None):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, 0)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    # attempt to infer class mapping from training generator saved mapping file (optional)
    if class_indices_path and os.path.exists(class_indices_path):
        import json
        with open(class_indices_path,"r") as f:
            mapping = json.load(f)
        inv = {v:k for k,v in mapping.items()}
        label = inv.get(idx, str(idx))
    else:
        label = str(idx)
    print("Predicted:", label, "Confidence:", float(np.max(preds)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer_finetuned.py <model_path> <image_path> [class_indices.json]")
    else:
        predict_image(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
