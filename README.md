# Enhancing Loan Default Prediction using High-Fidelity Synthetic Data (CTGAN)

## 📌 Project Overview
This project addresses the critical challenges of **Data Privacy** and **Severe Class Imbalance** in banking datasets. Using **Conditional Tabular GAN (CTGAN)**, we generated high-fidelity synthetic data to balance a loan default dataset (95:5 ratio) while preserving sensitive customer information.

## 🚀 Key Achievements
* **Recall Improvement:** Increased defaulter detection from **8.57% (Baseline)** to **11.38%** using the Train on Synthetic, Test on Real (TSTR) approach.
* **Data Quality:** Achieved a fidelity score of **85.38%**, validated via SDMetrics.
* **Class Balancing:** Successfully balanced a dataset of **~67,463 records** by generating 25,000 synthetic samples for the minority class.

---

## 🏗️ Technical Architecture & Working

The project implements a 5-step CTGAN pipeline specifically designed for mixed tabular data.

### 1. Data Preprocessing (Mode-Specific Normalization)
* Uses a **Variational Gaussian Mixture Model (VGM)** to handle continuous columns like `Annual Income`.
* Each value is represented as a cluster indicator and a scalar value to capture complex distributions.

### 2. Conditional Vector Generation
* Implements **Log-Frequency Sampling** to address class imbalance.
* It forces the model to sample the minority class (Defaulters) more frequently during training through a **Conditional Vector**.

### 3. Generator Network Structure
* **Residual Connections:** The generator ($G$) uses skip connections ($h_0 \oplus \dots$) to ensure the specific condition (e.g., "Make a Defaulter") is remembered throughout the network.
* **Mixed Outputs:** * Uses **tanh** for scalar values ($\hat{\alpha}$).
  * Uses **Gumbel-Softmax** for cluster modes ($\hat{\beta}$) and discrete categories ($\hat{d}$), ensuring the network remains differentiable for backpropagation.

### 4. Discriminator (PacGAN Architecture)
* **Packing:** To prevent **Mode Collapse**, the discriminator looks at a "Pack" of **10 rows** simultaneously ($\mathbf{r}_1 \dots \mathbf{r}_{10}$).
* **Stabilization:** Uses **Dropout** to prevent overfitting and **Leaky ReLU** (slope 0.2) to maintain gradient flow.
* **Output:** Provides a "Critic Score" representing the realism of the generated batch.

### 5. Loss Functions
* **WGAN-GP Loss:** Used for stabilizing the training of the GAN.
* **Cross-Entropy Loss:** Used to penalize the Generator if it fails to produce the specific class requested by the Conditional Vector.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** CTGAN (PyTorch), Neural Networks
* **Machine Learning:** Scikit-learn, Gradient Boosting
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, PCA (Principal Component Analysis)
* **Evaluation:** SDMetrics

---

## 📊 Evaluation Results
### 1. Fidelity (Similarity)
The synthetic data was compared against real data using PCA plots. The high overlap between real (Blue) and synthetic (Orange) distributions confirmed that statistical properties were preserved.

### 2. Utility (Effectiveness)
| Model Type | Recall (Defaulter Detection) |
| :--- | :--- |
| **Real Data (Baseline)** | 8.57% |
| **Synthetic Data (Balanced)** | **11.38%** |

---

## 📂 Folder Structure
* `Data/` - Contains the original and generated synthetic datasets.
* `Notebooks/` - Jupyter notebooks for training CTGAN and evaluating metrics.
* `Images/` - Architecture diagrams and PCA plots.
* `requirements.txt` - List of Python dependencies.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/loan-default-ctgan.git](https://github.com/your-username/loan-default-ctgan.git)
