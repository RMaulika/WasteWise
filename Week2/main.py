# Week2/main.py
import os, json
from src.dataset_loader import create_generators
from src.model_finetune import build_finetune_model
from src.train_model import train
from src.evaluate_model import evaluate_and_save
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DATA_ROOT = "../data" if os.path.exists("../data") else "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

def run_finetune(unfreeze_last_n=50, lr=1e-4, epochs=20, batch_size=32):
    train_gen, val_gen, test_gen = create_generators(TRAIN_DIR, VAL_DIR, TEST_DIR, image_size=(224,224), batch_size=batch_size)
    num_classes = len(train_gen.class_indices)
    print("Classes:", train_gen.class_indices)

    # compute class weights (optional)
    y = train_gen.classes
    classes = np.unique(y)
    cw_vals = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(enumerate(cw_vals))
    print("Class weights:", class_weight)

    model = build_finetune_model(num_classes=num_classes, input_shape=(224,224,3), unfreeze_last_n=unfreeze_last_n, lr=lr)
    history, ckpt = train(model, train_gen, val_gen, epochs=epochs, output_dir="Week2/outputs")

    # save class indices mapping for inference
    os.makedirs("Week2/outputs", exist_ok=True)
    with open("Week2/outputs/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    model.load_weights(ckpt)
    evaluate_and_save(model, test_gen, output_dir="Week2/outputs")

if __name__ == "__main__":
    # defaults: unfreeze 50 layers, lr 1e-4, epochs 20
    run_finetune(unfreeze_last_n=50, lr=1e-4, epochs=20, batch_size=32)
