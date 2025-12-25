## Brain Tumor Detection and Localization using CNN–LSTM with Grad-CAM

This project presents an explainable deep learning approach for automatic brain tumor detection, classification, and localization from MRI images using a CNN–LSTM hybrid model and Grad-CAM visualization.



## Project Overview

Brain tumor diagnosis using MRI scans is a critical and time-sensitive task. Manual analysis is prone to human error and requires expert radiologists. This project leverages deep learning to automate tumor classification while also providing visual explanations for model predictions.

The system classifies MRI images into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Grad-CAM is used to highlight tumor-affected regions, improving interpretability and trust in the model.



##  Objectives

- Automate brain tumor classification from MRI images
- Use CNN for spatial feature extraction
- Use LSTM for sequential feature learning
- Evaluate model performance using standard metrics
- Localize tumor regions using Grad-CAM
- Build an interpretable and reliable AI-based diagnostic system



## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- ROC–AUC Curve (multiclass)
- Training vs Validation Loss


## Grad-CAM Output

Grad-CAM generates a heatmap highlighting regions of the MRI scan that most influenced the model’s prediction.  
Red/yellow areas indicate tumor-relevant regions.



## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn


