Student: Maulika R
Duration: Oct 2025 – Nov 2025
Project Theme: Sustainability
Project Title: WasteWise – AI-Powered Organic vs Recyclable Waste Classification

⭐ 1. Introduction

Waste management plays a critical role in environmental sustainability. Proper segregation of waste at the source significantly reduces pollution, improves recycling efficiency, and minimizes landfill usage. However, manual sorting is inefficient, inconsistent, and prone to human error.

This project, WasteWise, uses Artificial Intelligence and Machine Learning to classify waste images into two categories:

Organic (O)

Recyclable (R)

The goal is to build a deployable AI model that assists in sustainable waste management and can be integrated into mobile or web applications.

⭐ 2. Problem Statement

Improper waste segregation causes environmental pollution, increases landfill load, and reduces recycling efficiency.
The challenge is to automate waste classification using AI so users can instantly identify the correct waste category.

⭐ 3. Objectives

Build an image classification model to distinguish Organic vs Recyclable waste.

Perform dataset preprocessing, augmentation, and train deep learning models.

Compare baseline, fine-tuned, and optimized models.

Convert the best model into TFLite for lightweight deployment.

Develop a Streamlit-based demo application.

Document weekly progress and produce final reports and presentations.

⭐ 4. Dataset Description

The dataset consists of images labeled into:

Organic (O)

Recyclable (R)

Dataset Split:

70% – Train

20% – Validation

10% – Test

Images were resized to 224×224, normalized, and augmented using transformations such as:

Rotation

Zoom

Horizontal flip

Brightness adjustments

⭐ 5. Methodology (Week-wise Work)
📌 Week 1 — Baseline Model

Dataset preparation and directory organization

Exploratory Data Analysis (EDA)

Implemented baseline model using MobileNetV2 with frozen layers

Achieved validation accuracy: 97.14%

Generated accuracy & loss plots

Created project structure and documentation

📌 Week 2 — Fine-Tuning

Unfroze top layers of MobileNetV2

Applied advanced augmentations

Hyperparameter tuning: learning rate, batch size, epochs

Improved validation performance

Saved updated plots, metrics, and model summary

📌 Week 3 — Experiments + Optimization

Ran two controlled experiments:
Experiment 1: unfreeze last 20 layers (LR = 5e-5)
Experiment 2: unfreeze last 50 layers (LR = 1e-4)

Compared performance using accuracy, loss, confusion matrices

Converted best model to TFLite

model_opt.tflite → 2.39 MB

model_int8.tflite → 2.58 MB

Created summary_table.csv of results

Built a Streamlit demo application

📌 Week 4 — Final Touches

Improved UI in Streamlit demo with meaningful labels and color-coded results

Added sidebar instructions & help for end-users

Captured demo screenshots

Created final report and PPT

Prepared GitHub repository for final submission

Created and tested final video demonstration

⭐ 6. Model Architecture

Base Model: MobileNetV2 (ImageNet pretrained)

Input Size: 224×224×3

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy, Precision, Recall, F1-Score

Fine-tuned Layers: Last 20/50 depending on experiment

⭐ 7. Results & Analysis
✔ Experiment Results
Experiment	Val Accuracy	Notes
Exp1	0.9813	20 layers unfrozen, LR=5e-5
Exp2	0.9790	50 layers unfrozen, LR=1e-4
TFLite	~same	size = 2.39–2.58MB
✔ Confusion Matrix Observations

High accuracy for Organic class

Slight performance variation for Recyclable (R) due to smaller dataset size

Model generalizes well overall

✔ Final Model Size (TFLite)

Float16 optimized model → 2.39 MB

INT8 quantized model → 2.58 MB

These sizes make it suitable for mobile deployment.

⭐ 8. Streamlit Demo Application

A simple user interface was created using Streamlit, allowing users to:

✔ Upload an image
✔ View the predicted class (Organic / Recyclable)
✔ See confidence score
✔ Test multiple images
✔ Read basic sustainability tips

Run the demo:
python -m streamlit run Week3/app.py


(Requires local copy of best_model_week2.h5 – not uploaded to GitHub for size reasons.)

⭐ 9. GitHub Repository Structure
Week1/   → Baseline model & EDA  
Week2/   → Fine-tuning code  
Week3/   → Experiments, TFLite models, Streamlit  
Week4/   → Final report, PPT, screenshots  
README.md → Project overview and instructions  
.gitignore → Ensures model files are not uploaded  

⭐ 10. Conclusion

The WasteWise project successfully demonstrates how AI can support sustainable waste management. The final model achieves high accuracy, efficient performance, and small size suitable for deployment. The Streamlit demo provides a user-friendly interface for testing waste classification in real time.

This project can be extended to classify more than two categories, integrate with a mobile app, or connect to smart bins for real-world deployment.

⭐ 11. Future Enhancements

Add more classes (paper, plastic, metal, glass, hazardous)

Deploy TFLite model in an Android app

Collect a more balanced dataset

Implement explainability (Grad-CAM for model visualization)

⭐ 12. References

TensorFlow Documentation

Keras Applications (MobileNetV2)

Streamlit Framework


⭐ 13. Screenshots

#### 1. Streamlit App Home Screen
![Streamlit Home](Week4/screenshots/screenshot1.png)

#### 2. Prediction Result
![Prediction](Week4/screenshots/screenshot2.png)

#### 3. Training & Experiment Outputs
![Experiment Plots](Week4/screenshots/screenshot3.png)
