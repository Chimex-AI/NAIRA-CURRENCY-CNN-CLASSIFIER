# Naira Currency Classification with CNNs (PyTorch)

A convolutional neural network that classifies images of Nigerian Naira banknotes into 7 denominations, built from scratch in PyTorch as part of an AI & Machine Learning exam project.

## 🎯 Project Overview

This project walks through the full applied ML pipeline for an image classification task on a small, real-world dataset:

- Loading and inspecting image data with `torchvision.datasets.ImageFolder`
- Computing dataset-specific normalization statistics (mean/std per channel)
- Building a baseline CNN from scratch
- Diagnosing class imbalance and fixing it with an oversampling strategy
- Adding data augmentation (flips, rotation, random resized crop)
- Building a deeper, regularized CNN with Batch Normalization
- Comparing baseline vs. improved model performance

## 🗂️ Dataset

- **7 classes:** ₦10, ₦20, ₦50, ₦100, ₦200, ₦500, ₦1000 notes
- **153 total images**, split into training, validation, and test sets
- Small dataset, so class balance and augmentation mattered a lot for generalization

## 🏗️ Models

| Model | Description |
|---|---|
| **Baseline CNN** | 2 conv blocks (16 → 32 filters) + fully connected layers |
| **Regularized CNN** | Deeper 5-conv-block architecture with Batch Normalization, trained on a class-balanced, augmented dataset |

## 📊 Results

| Model | Test Accuracy |
|---|---|
| Baseline CNN | 85.71% |
| Regularized CNN (balanced + augmented data) | **95.24%** |

The jump in accuracy highlights how much class balancing, augmentation, and batch normalization can matter on small, imbalanced datasets — even without changing the core problem or adding more raw data.

## 🛠️ Tech Stack

- Python, PyTorch, Torchvision
- scikit-learn (classification report, confusion matrix)
- Matplotlib (visualization)
- torchinfo (model summaries)

## 📁 Repository Contents

- `naira_cnn_classifier.ipynb` — full notebook with data exploration, model building, training, and evaluation

## 🚀 How to Run

1. Clone this repo
2. Install dependencies: `pip install torch torchvision matplotlib scikit-learn torchinfo`
3. Update the dataset path in the notebook to point to your local `Train`/`Test` folders
4. Run all cells in order

## 👤 Author

Eruba Echika Chimezirim
