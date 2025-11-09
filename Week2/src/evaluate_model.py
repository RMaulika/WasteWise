# Week2/src/evaluate_model.py
import numpy as np, os, matplotlib.pyplot as plt, csv
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_and_save(model, test_gen, output_dir="Week2/outputs", top_mis=10):
    os.makedirs(output_dir, exist_ok=True)
    steps = test_gen.samples // test_gen.batch_size + 1
    preds = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    # classification report & CSV
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    with open(os.path.join(output_dir, "metrics_week2.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["class","precision","recall","f1-score","support"])
        for cls in labels:
            r = report.get(cls, {})
            writer.writerow([cls, r.get("precision"), r.get("recall"), r.get("f1-score"), r.get("support")])
        # overall
        writer.writerow(["accuracy", report.get("accuracy")])

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix_week2.png"))

    # save a few misclassified examples (optional: uses test_gen.filepaths, available in Keras DirectoryIterator)
    try:
        filepaths = test_gen.filepaths
        mis_idx = [i for i,(t,p) in enumerate(zip(y_true,y_pred)) if t!=p][:top_mis]
        mis_dir = os.path.join(output_dir, "misclassified")
        os.makedirs(mis_dir, exist_ok=True)
        import shutil
        for i, idx in enumerate(mis_idx):
            src = filepaths[idx]
            dst = os.path.join(mis_dir, f"mis_{i}_{os.path.basename(src)}")
            shutil.copy(src, dst)
    except Exception as e:
        print("Could not save misclassified examples:", e)
