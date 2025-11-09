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