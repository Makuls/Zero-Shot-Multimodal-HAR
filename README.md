# ⌚ The Limits of Edge-AI Compression: Zero-Shot Cross-Dataset Generalization in Multimodal HAR

**Author:** Makul Swami | **Institution:** IIT Patna (M.Tech in AI & Data Science)

## 📌 Abstract
This repository contains the official codebase for an ablation study investigating the architectural limits of deploying Multimodal Human Activity Recognition (HAR) models to hyper-compressed edge devices (e.g., smartwatches). 

The research establishes a foundational 27.4M-parameter Hierarchical Multimodal Teacher (Vision + IMU) and systematically evaluates the boundaries of compressing this intelligence into a 2.3M-parameter Edge Student. Through rigorous zero-shot cross-dataset evaluation (training on UTD-MHAD, testing on MMAct), this project empirically proves that while heavy foundational models can survive severe domain shifts, edge-optimized models suffer from catastrophic representational collapse regardless of the compression strategy utilized.

## 🏗️ Model Architecture
* **Master Teacher (~27.4M Parameters):** * *Sensor Backbone:* Custom Hierarchical Masked Autoencoder (Transformer) pre-trained via Self-Supervised Learning (SSL) to reconstruct masked 6-axis IMU data.
  * *Vision Backbone:* ResNet50 (stripped of classification head) for spatial/posture embedding.
  * *Fusion:* Dynamic Late-Fusion MLP.
* **Edge Student (~2.3M Parameters):**
  * *Sensor Backbone:* 1D-CNN.
  * *Vision Backbone:* MobileNetV2.

## 🔬 Experimental Phases & Key Discoveries

### Phase 1: The Foundational Baseline (Success)
The heavy Master Teacher was tested under strict zero-shot conditions on the unseen MMAct dataset. 
* **Result:** Achieved an **Aligned Semantic Accuracy of 26.38%** (7x higher than the mathematical random baseline of 3.7%). The model successfully retained multi-modal feature entropy and distributed variance across distinct activity boundaries.

### Phase 2: Standard Knowledge Distillation (Failure)
Compressing the Teacher into the Student using standard logits distillation.
* **Result:** **Mode Collapse (Class 19).** The low-capacity student memorized the source domain's background wall. Upon facing domain shift, it suffered catastrophic representational failure.

### Phase 3: Adversarial Unsupervised Domain Adaptation / UDA (Failure)
Implementing a Gradient Reversal Layer (GRL) to force the Student to unlearn domain-specific features.
* **Result:** **Feature Erasure (Mode Collapse to Class 23).** The 2.3M parameter model lacked the capacity to simultaneously balance adversarial domain constraints and complex physical action recognition.

### Phase 4: Intermediate Feature Distillation (Failure)
Forcing the Student to mathematically mimic the Teacher's deep 512-dimensional internal feature map.
* **Result:** **Representational Bottleneck.** The shallow convolutional kernels of the edge model could not map the topological complexity of the Teacher's attention mechanisms.

### Phase 5: Domain-Blind Data Augmentation (Failure)
Applying aggressive Random Erasing, Color Jitter, and IMU Axis Permutation to force the edge model to ignore background memorization.
* **Result:** **Network Underfitting (6.94% Validation Accuracy).** Erasing the background exposed the hard capacity limit of the edge model, proving it mathematically incapable of learning complex physical geometry through heavy noise.

## 📊 Conclusion
This research proves that Zero-Shot Cross-Dataset Generalization is currently unfeasible for hyper-compressed multimodal edge models. Deploying HAR algorithms to wearable devices strictly requires Target-Domain Fine-Tuning (Supervised Few-Shot Learning) on the specific deployment hardware.

## 📂 Repository Structure
```text
HAR-Edge-Compression/
├── data/                   # (Ignored) Raw UTD-MHAD and MMAct datasets
├── saved_models/           # (Ignored) .pth weights for Teacher and Student
├── src/
│   ├── datasets/           # Dataloaders, Vision/IMU Scramblers, SSL masking
│   ├── models/             # Transformer, ResNet, MobileNet, CNN definitions
│   └── utils/              # Plotting scripts, accuracy calculators
├── train_multimodal_distillation.py
├── train_uda_distillation.py
├── train_feature_distillation.py
├── evaluate_ablation_student.py
├── calculate_teacher_accuracy.py
└── README.md