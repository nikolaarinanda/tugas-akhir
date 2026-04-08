# TikTok Comment Sentiment Analysis Using TextCNN

<!-- This is the short explanation about the title of the project. -->

## Project Overview

This project is a Final Project that implements a TextCNN to perform sentiment analysis on TikTok comments in Indonesian language. The system is designed to classify comments into two categories: Cyberbullying (insult/embarrass content) and Non-Cyberbullying (normal/clear content).

**Institution**: Institut Teknologi Sumatera (ITERA)
**Study Program**: Informatic Engineering
**Author**: Nikola Arinanda
**Year**: 2026

---

## Abstract

This project uses TextCNN architecture to analyze YouTube comments sentiment. Initially, the data will go through preprocessing stages such as data division (80:20), case folding, text cleaning, augmentation (AEDA, random swap character, random delete character), tokenization and stopword removal to prepare the data. The next step is model training using k-fold cross-validation as many as 5 fold to ensure robustness and good generalization. Final step is model evaluation using confussion matrix such as accuracy, precision, recall and F1 score.

---

## Project Structure

<!-- <pre> -->

```bash
tugas-akhir-main/
├── dataset/
│ ├── k_fold.json           # k-fold cross-validation dictionary
│ └── cyberbullying.csv     # Original dataset
├── code/
│ ├── datareader.py         # Data loader and preprocessing
│ ├── model.py              # Model architecture
│ └── train.py              # Main script for model training
├── model_outputs
│ ├── run_YYYYMMDD_HHMMSS/
│ │ ├── fold_1_model.pth    # Model output
│ │ ├── fold_2_model.pth    # ...
│ │ ├── fold_3_model.pth
│ │ └── fold_4_model.pth
│ │ └── fold_5_model.pth
│ └── ...
└── report/
│ └── thesis.pdf            # Documentation and reports
└── requirements.txt        # Python dependencies
```

<!-- </pre> -->

---

## Environment Setup

### Prerequisites

This project requires:

- Python: 3.8 or higher (tested with Python 3.9+)
- CUDA: Optional (for GPU acceleration)

### System Requirements

- RAM: Minimum 8 GB (recommended 16 GB)
- Storage: Minimum 10 GB (for model and dataset)
- GPU: Optional, but highly recommended for faster training

---

## Dependencies

All dependencies are listed in the `requirements.txt` file. Main libraries:

```bash
| Library        | Version  | Purpose                              |
|----------------|----------|--------------------------------------|
| torch          | >=2.0.0  | Deep learning framework              |
| pandas         | >=1.5.0  | Data manipulation                    |
| numpy          | >=1.23.0 | Numerical computing                  |
| matplotlib     | >=3.7.0  | Data visualization                   |
| seaborn        | >=0.12.0 | Statistical visualization            |
| scikit-learn   | >=1.2.0  | Machine learning utilities           |
| transformers   | >=4.30.0 | NLP models (IndoBERT, etc.)          |
| nltk           | >=3.8.0  | Text preprocessing                   |
| tqdm           | >=4.65.0 | Progress bar                         |
| wandb          | >=0.15.0 | Experiment tracking                  |
```

For the complete list, see `requirements.txt`

---

## Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/nikolaarinanda/tugas-akhir.git
cd tugas-akhir
```

### Step 2: Create Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.
**Using venv (built-in python)**:

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Using conda**:

```bash
conda create -n youtube-sentiment python=3.9
conda activate youtube-sentiment
```

### Step 3: Install Dependencies

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Note for PyTorch with GPU**: If you want to use GPU, install the CUDA-specific version of PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Run Preprocessing

`py preprocessing.py`

### Step 5: Run Train

`py train.py`

---

<!-- This research focuses on TextCNN and modified SEDepthwiseTextCNN for classifying cyberbullying in the TikTok comments dataset. In this study, I improve the accuracy with model modified SEDepthwiseTextCNN compared to previous [research](https://ieeexplore.ieee.org/document/10468424) using same dataset that uses BERT. -->

For using LaTeX on a desktop computer with VS Code, you can follow [this](https://youtu.be/4lyHIQl4VM8?si=TOYXOIaCTGxaEusH) video.
