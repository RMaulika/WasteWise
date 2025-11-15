# WasteWise - AI/ML Internship Project (Sustainability Theme)

This project focuses on classifying waste into **Organic (O)** and **Recyclable (R)** categories using deep learning (MobileNetV2).  
It is part of the **Shell–Edunet Skills4Future Internship (Oct–Nov 2025)** under the **Sustainability** theme.

----

## 📁 Project Structure
Waste-wise/
│
├── Week1/ # Baseline model (MobileNetV2)
│ ├── src/
│ ├── outputs/
│ ├── main.py
│ └── sample_images/
│
├── dataset/ # Original dataset (kept local)
├── data/ # Split dataset (train/val/test)
├── Week2/ # (Will contain fine-tuned model & augmentation)
├── README.md
└── requirements.txt

---

## 🧠 Week 1 – Baseline Model

**Goal:** Build a baseline classifier using MobileNetV2 for binary waste classification.  

**Dataset:**  
- Classes: *Organic (O)* and *Recyclable (R)*
- Dataset organized into train/validation/test using a custom split script.

**Command to run:**
```bash
cd Week1
python main.py

**Outputs:**
outputs/best_model.h5
outputs/accuracy_plot.png
outputs/loss_plot.png

Week 1 Summary:
✅ Dataset setup completed
✅ EDA and baseline model trained (MobileNetV2)
✅ Validation accuracy = 97.14%

---

## 🗓️ Week 2 – Fine-Tuning & Data Augmentation

** Objective **
Improve the baseline MobileNetV2 model’s performance by:
- Fine-tuning deeper layers  
- Applying image data augmentation  
- Evaluating post-tuning accuracy and loss

** Steps Performed **
✅ Loaded preprocessed dataset from Week 1  
✅ Implemented data augmentation using `ImageDataGenerator` (rotation, zoom, flips)  
✅ Unfrozen top layers of MobileNetV2 and fine-tuned with a lower learning rate  
✅ Trained the fine-tuned model for multiple epochs  
✅ Evaluated and saved updated metrics and plots  

**Command to Run:**
```bash
python Week2/main.py

**Outputs:**
Week2/outputs/fine_tuned_model.h5
Week2/outputs/accuracy_plot_week2.png
Week2/outputs/loss_plot_week2.png
Week2/outputs/confusion_matrix_week2.png

Week 2 Summary:
✅Validation accuracy (after fine-tuning): ≈ 98–99 %
✅Noticeable reduction in validation loss and improved generalization

---

## Week 3 — Experiments, TFLite conversion & Demo

**Activities**
- Performed two fine-tuning experiments on MobileNetV2:
  - `exp1`: unfreeze last 20 layers, lr=5e-5, epochs=12 — validation accuracy = **0.98137**
  - `exp2`: unfreeze last 50 layers, lr=1e-4, epochs=12 — validation accuracy = **0.97905**
- Converted the best model to TFLite:
  - `model_opt.tflite` (post-training quant) **2.39 MB**
  - `model_int8.tflite` (full int8 quant) **2.58 MB**
- Built a simple Streamlit demo (`Week3/app.py`) to run inference locally with the saved Keras model (kept locally, not pushed).
- Saved experiment plots, confusion matrices and metrics in `Week3/outputs/exp1` and `Week3/outputs/exp2`.
- Created `Week3/outputs/summary_table.csv` comparing experiments and model sizes.

**Files / folders (important)**
- `Week3/src/convert_to_tflite.py` — conversion script (post-training & int8).
- `Week3/src/tflite_infer.py` — small script to run inference with the `.tflite`.
- `Week3/app.py` — Streamlit demo (requires local `Week2/outputs/best_model_week2.h5`).
- `Week3/outputs/` — plots, metrics and TFLite files (`model_opt.tflite`, `model_int8.tflite`).
- `Week2/outputs/best_model_week2.h5` — **local only** (excluded from Git).

**How to run demo (locally)**
```powershell
# activate venv
.\.venv\Scripts\activate

# run streamlit demo (uses local .h5 model)
python -m streamlit run Week3/app.py
