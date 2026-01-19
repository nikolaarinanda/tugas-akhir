#!/usr/bin/env python
# coding: utf-8

# ## 1. Download Packages & Resource
# Download packages yang diperlukan untuk preprocessing, model ataupun training.

# In[ ]:


# FOR SOME SECTION
import torch
import os

# FOR PREPROCESSING SECTION
import re
import json
import nltk
import random
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

# FOR MODEL SECTION
from torch import nn # from torch import torch.nn as nn
import torch.nn.functional as F

# FOR TRAIN SECTION
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt

# FOR EVALUATION SECTION
import seaborn as sns

from tqdm import tqdm, trange
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW, Muon
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


# In[ ]:


# Download NLTK yang diperlukan
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('indonesian'))


# ## 2. Dataset Class
# Kelas dari dataset yang mana dilakukan proses preprocessing.

# In[ ]:


class CyberbullyingDataset(Dataset):
    def __init__(
            self,
            file_path="../dataset/cyberbullying.csv",
            tokenizer_name="indobenchmark/indobert-base-p1",
            folds_file="k_folds.json",
            random_state=29082002,
            split="train",
            fold=0,
            n_folds=5,
            max_length=128,
    ):
        self.file_path = file_path
        self.folds_file = folds_file
        self.random_state = random_state
        self.split = split
        self.fold = fold
        self.n_folds = n_folds
        self.max_length = max_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = len(self.tokenizer)

        # Load dataset
        self.load_data()
        # Setup n-Fold Cross Validation
        self.setup_folds()
        # Mempersiapkan Indices (bentuk jamak index)
        self.setup_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Mengambil index dari data yang akan diambil
        idx = self.indices[idx]
        # Mengambil data komentar dan sentiment
        text = str(self.df.iloc[idx]["comment"])
        label = str(self.df.iloc[idx]["sentiment"])
        # Melakukan Pre-Processing
        comment_processed = self.preprocess(text)
        # Tokenisasi
        encoding = self.tokenizer(
            comment_processed,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(label), dtype=torch.long),
            'original_text': text,
            'processed_text': comment_processed,
            'original_index': idx
        }
        return data

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

        # Load dictionary slang dari file JSON
        with open('slang_dictionary.json', 'r') as file:
            slang_dict = json.load(file)

    def aeda_augment(self, text):
        """
        Melakukan augmentasi teks dengan metode AEDA:
        Menyisipkan tanda baca secara acak di posisi acak dalam teks.
        """
        punctuations = [".", ";", "?", ":", "!", ","]
        words = text.split()
        if len(words) == 0:
            return text

        # # Tentukan berapa banyak tanda baca yang akan disisipkan
        # n_insert = random.randint(1, max(1, len(words) // 3))

        # # Pilih posisi acak untuk sisipan
        # positions = random.sample(range(len(words)), n_insert)
        # positions.sort(reverse=True)  # disisipkan dari belakang biar indeks tidak bergeser

        # for pos in positions:
        #     punct = random.choice(punctuations)
        #     words.insert(pos, punct)

        # Pilih posisi acak untuk sisipan
        position = random.randint(0, len(words) - 1)

        # Pilih tanda baca acak
        punct = random.choice(punctuations)

        # Sisipkan tanda baca ke dalam list kata
        words.insert(position, punct)

        return " ".join(words)

    def random_typo(self, text):
        words = text.split()
        if len(words) < 1:
            return text
        # Pilih satu kata secara acak untuk dimodifikasi
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_list = list(word)
            # Pilih posisi acak untuk swap
            i = random.randint(0, len(char_list) - 2)
            # swap 2 huruf berdekatan
            char_list[i], char_list[i+1] = char_list[i+1], char_list[i] 
            words[idx] = ''.join(char_list)
        return ' '.join(words)

    def random_swap(self, text):
        words = text.split()
        if len(words) < 2:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def random_delete(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        # Pilih satu kata secara acak untuk dihapus
        idx = random.randint(0, len(words) - 1)
        # Hapus kata tersebut
        del words[idx]
        return ' '.join(words)

    def augmentation_text(self, text, probability=0.5):
        # Hanya lakukan augmentasi dengan probabilitas tertentu
        if random.random() > probability:
            return text

        # PROBABILITAS ACAK
        # Daftar semua fungsi augmentasi yang tersedia
        augmentations = [
            self.aeda_augment,
            self.random_typo,
            self.random_swap,
            self.random_delete
        ]

        # Pecah kalimat menjadi kumpulan kata
        words = text.split()
        # Tentukan berapa banyak augmentasi yang akan dilakukan
        n_insert = random.randint(1, max(1, len(words) // 3))
        for i in range(n_insert):
            # Pilih satu augmentasi secara acak
            augmentation_func = random.choice(augmentations)
            # Terapkan augmentasi yang dipilih
            text = augmentation_func(text)
        return text

        # # Pilih satu augmentasi secara acak
        # augmentation_func = random.choice(augmentations)
        # # Mengembalikan augmentasi yang dipilih
        # return augmentation_func(text)

    def preprocess(self, text):
        # Konversi ke huruf kecil
        text = text.lower()

        # Hapus mention (@) dan hashtag (#)
        text = re.sub(r'@\w+|#\w+', '', text)

        # # Hapus emoji dan karakter non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Augmentasi
        text = self.augmentation_text(text)

        # Tokenisasi
        words = nltk.word_tokenize(text)

        # Menghapus stopwords
        words = [word for word in words if word not in stop_words]

        # Menggabungkan kembali kata-kata menjadi kalimat
        text = ' '.join(words)

        return text

    def setup_indices(self):
        '''
        Mempersiapkan indices untuk data yang akan di-training
        '''
        fold_key = f"fold_{self.fold}"
        if self.split == "train":
            self.indices = self.fold_indices[fold_key]['train_indices']
        else:
            self.indices = self.fold_indices[fold_key]['val_indices']

    def setup_folds(self):
        # Jika fold sudah ada, maka load fold
        if os.path.exists(self.folds_file):
            self.load_folds()
        # Jika tidak ada, maka buat fold
        else:
            self.create_folds()

    def load_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
        self.fold_indices = fold_data['fold_indices']
        print(f"Menggunakan {fold_data['n_folds']} folds dengan {fold_data['n_samples']} samples")

    def create_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
        print(f"Membuat n-fold CV dengan random state {self.random_state}")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # print("\nStratified k-fold positif samples per fold:")
        # for _, val_idx in skf.split(self.df, self.df['sentiment']):
        #     print(f"{np.sum(self.df['sentiment'].iloc[val_idx] == 1)} out of {len(val_idx)}")

        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['sentiment'])):
            fold_indices[f"fold_{fold}"] = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            }

        # Simpan fold ke file
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices': fold_indices,
                'n_samples': len(self.df),
                'n_folds': self.n_folds,
                'random_state': self.random_state
            }, f)

            self.fold_indices = fold_indices
            print(f'Created {self.n_folds}-fold indices and saved to {self.folds_file}')

    def load_data(self):
        print(f'Loading data from {self.file_path}...')
        self.df = pd.read_csv(self.file_path) # Load csv
        # self.df.columns = ['sentiment', 'comment'] # Rename columns
        if len(self.df.columns) == 2:
            self.df.columns = ['sentiment', 'comment']
        else:
            print("⚠️ Jumlah kolom tidak sesuai, kolom asli:", self.df.columns)
        self.df = self.df.dropna(subset=['sentiment', 'comment']) # Drop NaN values
        self.df['sentiment'] = self.df['sentiment'].astype(int) # Convert sentiment to int
        self.df['sentiment'] = self.df['sentiment'].apply(lambda x: 1 if x == -1 else 0) # Transform labels: convert -1 to 1, and 1 to 0
        self.df = self.df[(self.df['sentiment'] == 0) | (self.df['sentiment'] == 1)] # Filter sentiment

        # Undersampling menyeimbangkan dataset (hanya untuk split "train")
        if self.split == "train":
            df_label_0 = self.df[self.df['sentiment'] == 0]
            df_label_1 = self.df[self.df['sentiment'] == 1]

            min_samples_per_class = min(len(df_label_0), len(df_label_1))

            df_label_0_undersampled = df_label_0.sample(n=min_samples_per_class, random_state=self.random_state)
            df_label_1_undersampled = df_label_1.sample(n=min_samples_per_class, random_state=self.random_state)

            self.df = pd.concat([df_label_0_undersampled, df_label_1_undersampled])
            self.df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # df_label_0 = self.df[self.df['sentiment'] == 0]
        # df_label_1 = self.df[self.df['sentiment'] == 1]

        # min_samples_per_class = min(len(df_label_0), len(df_label_1))

        # df_label_0_undersampled = df_label_0.sample(n=min_samples_per_class, random_state=self.random_state)
        # df_label_1_undersampled = df_label_1.sample(n=min_samples_per_class, random_state=self.random_state)

        # self.df = pd.concat([df_label_0_undersampled, df_label_1_undersampled])
        # self.df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)


# ### 2x. Main Section Preprocessing
# Bagian utama untuk menjalankan program.

# In[ ]:


# if __name__ == "__main__":
#     dataset = CyberbullyingDataset(fold=0, split="train")
#     data = dataset[0]
#     print(data)


# ## 3. Model Class
# Kelas dari model machine learning yang akan di training.

# In[ ]:


# Conv1DFlat (Muon-compatible)    
class Conv1DFlat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 2D parameter (Muon-compatible)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels * kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # inisialisasi mirip Conv1D
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        weight_3d = self.weight.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size
        )
        return F.conv1d(x, weight_3d, self.bias)

# TextCNN
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_channels=300,
        num_classes=2,
        conv_filters=100,
        kernel_sizes=[3, 4, 5],
        dropout_rate=0.5
    ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, in_channels)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, conv_filters, kernel_size=k)
            # Conv1DFlat(in_channels, conv_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(conv_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))        # (batch, conv_filters, L')
            h = torch.max(h, dim=2)[0]
            conv_outs.append(h)

        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# 🔧 TextCNN ResNorm
class ResidualTextCNN(nn.Module):
    """
    ResidualTextCNN
    - Gabungan TextCNNLight + BatchNorm + Residual connection
    - Stabil, ringan, cocok untuk teks pendek seperti komentar media sosial.
    """

    def __init__(self, vocab_size, in_channels=100, num_classes=2, conv_filters=100, dropout_rate=0.5):
        super(ResidualTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_channels, padding_idx=0)

        # Dua layer konvolusi dengan ukuran kernel berbeda
        self.conv1 = nn.Conv1d(in_channels, conv_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, conv_filters, kernel_size=4, padding=1)

        # Normalisasi batch setelah konvolusi
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.bn2 = nn.BatchNorm1d(conv_filters)

        # Shortcut projection agar dimensi sama (residual)
        self.shortcut = nn.Linear(in_channels, conv_filters * 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(conv_filters * 2, num_classes)

    def forward(self, x):
        # 1️⃣ Embedding
        x_embed = self.embedding(x)  # (batch, seq_len, in_channels)
        x_embed_t = x_embed.permute(0, 2, 1)  # (batch, in_channels, seq_len)

        # 2️⃣ Convolution + ReLU + BatchNorm + Global Max Pooling
        x1 = F.relu(self.bn1(self.conv1(x_embed_t))).max(dim=2)[0]
        x2 = F.relu(self.bn2(self.conv2(x_embed_t))).max(dim=2)[0]

        # 3️⃣ Concatenate hasil konvolusi
        x_cat = torch.cat((x1, x2), dim=1)  # (batch, conv_filters * 2)

        # 4️⃣ Residual connection: proyeksikan embedding ke dimensi yang sama
        residual = self.shortcut(x_embed.mean(dim=1))  # rata-rata embedding → dim (conv_filters*2)
        x_res = x_cat + residual  # tambah residual shortcut

        # 5️⃣ Dropout dan FC
        x_res = self.dropout(x_res)
        out = self.fc(x_res)
        return out

# SE (Squeeze-and-Excitation) Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x shape: (batch_size, channels)

        # SQUEEZE:
        # Pada TextCNN, dimensi temporal sudah dihilangkan
        # oleh global max pooling sebelumnya.
        # Oleh karena itu, tidak diperlukan lagi operasi
        # pooling tambahan (mean / avg).
        # Representasi vektor ini sudah merupakan ringkasan
        # global dari tiap channel untuk satu sampel.

        # EXCITATION:
        # Dua fully connected layer digunakan untuk
        # mempelajari dependensi antar channel dan
        # menghasilkan bobot pentingnya masing-masing channel.
        # Sigmoid memastikan bobot berada pada rentang [0, 1].
        w = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))

        # Channel-wise reweighting:
        # Setiap channel diperkuat atau dilemahkan
        # secara adaptif berdasarkan konteks input.
        return x * w

# SEDepthwise TextCNN
class SEDepthwiseTextCNN(nn.Module):
    def __init__(self, vocab_size, in_channels=300, num_classes=2, conv_filters=100, kernel_sizes=[3,4,5], dropout_rate=0.5):
        super(SEDepthwiseTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_channels, padding_idx=0)

        # Depthwise + Pointwise convolution blocks
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=ks, groups=in_channels, padding=0),  # depthwise
                nn.Conv1d(in_channels, conv_filters, kernel_size=1),  # pointwise
                nn.ReLU()
            )
            for ks in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.se_block = SEBlock(conv_filters * len(kernel_sizes))
        self.fc = nn.Linear(conv_filters * len(kernel_sizes), num_classes)

    # Helper block untuk konvolusi + aktivasi + pooling
    def conv_block(self, x, depthwise, pointwise):
        x = depthwise(x)
        x = F.relu(pointwise(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)   # (batch, seq_len, in_channels)
        x = x.permute(0, 2, 1)  # (batch, in_channels, seq_len)

        conv_outputs = []
        for conv in self.convs:
            y = conv(x)
            y = F.max_pool1d(y, kernel_size=y.size(2)).squeeze(2)
            conv_outputs.append(y)

        x_cat = torch.cat((conv_outputs), dim=1)    # (batch, conv_filters * 3)
        x_cat = self.se_block(x_cat)
        x_cat = self.dropout(x_cat)

        return self.fc(x_cat)

# TextCNN Enhanced
class EnhancedTextCNN(nn.Module):
    def __init__(self, vocab_size, in_channels=100, num_classes=2, conv_filters=100, dropout_rate=0.5):
        super(EnhancedTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_channels, padding_idx=0)

        # Depthwise + Pointwise convolution blocks
        self.depthwise_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, dilation=1, groups=in_channels, padding=1)
        self.pointwise_conv1 = nn.Conv1d(in_channels, conv_filters, kernel_size=1)

        self.depthwise_conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=4, dilation=2, groups=in_channels, padding=3)
        self.pointwise_conv2 = nn.Conv1d(in_channels, conv_filters, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)
        self.se_block = SEBlock(conv_filters * 2)
        self.fc = nn.Linear(conv_filters * 2, num_classes)

    # Helper block untuk konvolusi + aktivasi + pooling
    def conv_block(self, x, depthwise, pointwise):
        x = depthwise(x)
        x = F.relu(pointwise(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)            # (batch, seq_len, in_channels)
        x = x.permute(0, 2, 1)           # (batch, in_channels, seq_len)

        # 🔸 Dua jalur konvolusi dengan cara yang sama
        x1 = self.conv_block(x, self.depthwise_conv1, self.pointwise_conv1)
        x2 = self.conv_block(x, self.depthwise_conv2, self.pointwise_conv2)

        # 🔸 Gabung dan lanjut ke SE + FC
        x_cat = torch.cat((x1, x2), dim=1)
        x_cat = self.se_block(x_cat)
        x_cat = self.dropout(x_cat)
        out = self.fc(x_cat)
        return out


# ## 4. TRAIN SECTION
# Bagian untuk train model yang sudah dibuat.

# In[ ]:


# Reproducibility
SEED = 29082002
# Training Model
DATASET_PATH = '../dataset/cyberbullying.csv'
MODEL_OUTPUT_PATH = 'model_outputs'
# K-fold Cross-validation
N_FOLDS = 5
MAX_LENGTH = 128
# Training Model
EPOCHS = 100
BATCH_SIZE = 50 # ablation: (25, @50, 100)
LEARNING_RATE = 5e-4 # ablation: (5e-2, @5e-3, 5e-4)
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1'
OPTIMIZER_NAME = 'Muon' # ablation: (AdamW, Muon)
# Model Hyperparameters
NUM_CLASSES = 2
EMBEDDING_DIM = 100 # light=100, medium=@300, large=600 => ablation: (50 , 100, 200)
KERNEL_SIZE = [3, 4] # light=[3, 4], medium=@[3, 4, 5], large=[3, 4, 5, 6] => ablation: (@[3], [3, 4], @[3,4,5])
CONV_FILTERS = 100 # light=50, medium=@100, large=200 => ablation: (25, 50, @100)
DROPOUT_RATE = 0.5 # ablation: (0.4, @0.5, 0.6)
# Early Stopping
PATIENCE = 10
# Notebook
IS_NOTEBOOK = True
# WANDB_NOTE = 'Hyperparameter baseline untuk TextCNN SEDepthwise'
WANDB_NOTE = 'Light TextCNN best hyperparameter tuning'
WANDB_GROUP = 'Light TextCNN best hyperparameter tuning'


# ### 4.1. For One Fold
# Untuk train 1 fold.

# In[ ]:


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CNN for sentiment analysis')

    # Seed and reproducibility
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')

    # Data parameters
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, 
                        help='Path to dataset file')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                        help='Maximum sequence length')

    # Model parameters
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME,
                        help='Tokenizer name')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for embedding')
    parser.add_argument('--embed_dim', type=int, default=EMBEDDING_DIM, 
                        help='Embedding dimension for CNN')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, 
                        help='Number of classes')
    parser.add_argument('--conv_filters', type=int, default=CONV_FILTERS, 
                        help='Number of filters for CNN')
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE, 
                        help='Kernel sizes for CNN')

    # Training parameters
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help='Fold number for cross-validation')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')

    # Output parameters
    parser.add_argument('--output_model', action='store_false', default=True,
                       help='Save model after training')

    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_PATH,
                        help='Directory to save model outputs')

    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_group', type=str, default=WANDB_GROUP,
                    help='Create group for Weights & Biases runs')
    parser.add_argument('--wandb_note', type=str, default=WANDB_NOTE,
                        help='Add Weights & Biases notes')

    # Early Stopping parameters
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping (epochs to wait after no improvement)')

    args, unknown = parser.parse_known_args()
    return args

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_output_dir(base_dir):
    """Create timestamped output directory"""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_dataloaders_for_fold(args, fold=0):
    """Create train and validation datasets/dataloaders for the fold"""
    # Membuat fold train dataset
    train_dataset = CyberbullyingDataset(
        file_path=args.dataset_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed,
        split="train",
        n_folds=args.n_folds,
        fold=fold,
        max_length=args.max_length,
    )

    # Membuat fold val dataset
    val_dataset = CyberbullyingDataset(
        file_path=args.dataset_path,
        tokenizer_name=args.tokenizer,
        random_state=args.seed,
        split="val",
        n_folds=args.n_folds,
        max_length=args.max_length,
    )

    # Membuat DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    return train_loader, val_loader

def save_conf_matrix(true_labels, pred_labels):
    df = pd.DataFrame({
        "y_true": true_labels,
        "x_pred": pred_labels
    })

    cm = confusion_matrix(df['y_true'], df['x_pred'])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_df,
        # annot=labels,
        annot=True,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    fig.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # plt.show()

def create_model(args, train_loader, model_name="default-textcnn"):
    if model_name == 'default-textcnn':
        model = TextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=args.embed_dim,
            conv_filters=args.conv_filters,
            kernel_sizes=args.kernel_size,
            dropout_rate=args.dropout
        )

    elif model_name == 'light-textcnn':
        # print("Model vocabulary size:", train_loader.dataset.vocab_size)
        model = TextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=100,
            conv_filters=50,
            kernel_sizes=[3, 4],
            dropout_rate=0.5
        )

    elif model_name == 'medium-textcnn':
        model = TextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=300,
            conv_filters=100,
            kernel_sizes=[3, 4, 5],
            dropout_rate=0.5
        )

    elif model_name == 'weight-textcnn':
        model = TextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=600,
            conv_filters=200,
            kernel_sizes=[3, 4, 5, 6],
            dropout_rate=0.5
        )

    elif model_name == 'residual-textcnn':
        model = ResidualTextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=args.embed_dim,
            conv_filters=args.conv_filters,
            dropout_rate=args.dropout
        )

    elif model_name == 'sedepthwise-textcnn':
        model = SEDepthwiseTextCNN(
            vocab_size=train_loader.dataset.vocab_size,
            in_channels=args.embed_dim,
            conv_filters=args.conv_filters,
            kernel_sizes=args.kernel_size,
            dropout_rate=args.dropout
        )

    return model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    labels_all, preds_all = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        total_loss += loss.item()
        total += batch['labels'].size(0)
        correct += (preds == batch['labels']).sum().item()

        labels_all.extend(batch['labels'].cpu().tolist())
        preds_all.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100 * correct / total,
        "labels": labels_all,
        "preds": preds_all
    }

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    labels_all, preds_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = to_device(batch, device)
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])

            _, preds = torch.max(outputs, 1)

            total_loss += loss.item()
            total += batch['labels'].size(0)
            correct += (preds == batch['labels']).sum().item()

            labels_all.extend(batch['labels'].cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100 * correct / total,
        "labels": labels_all,
        "preds": preds_all
    }

def compute_metrics(labels, preds):
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
        # "precision": precision_score(labels, preds, average='weighted', zero_division=0),
        # "recall": recall_score(labels, preds, average='weighted', zero_division=0),
        # "f1": f1_score(labels, preds, average='weighted', zero_division=0)
    }

def cnn_train(args, output_dir="none", cur_fold=0, model_name='light'):
    print(f"\n{'='*5} Fold {cur_fold} {'='*5}")

    device = get_device()
    train_loader, val_loader = get_dataloaders_for_fold(args, cur_fold)

    model = create_model(args, train_loader, model_name).to(device)
    if OPTIMIZER_NAME == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif OPTIMIZER_NAME == 'Muon':
        optimizer = Muon(
            [p for p in model.parameters() if p.dim() == 2],
            lr=args.lr
        )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    early_stopping = EarlyStopping(patience=PATIENCE)

    best_val_loss = float("inf")
    best_val_labels, best_val_preds = None, None
    best_epoch = None

    for epoch in range(args.epochs):
        train_out = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_out = validate_one_epoch(model, val_loader, criterion, device)

        train_metrics = compute_metrics(train_out["labels"], train_out["preds"])
        val_metrics = compute_metrics(val_out["labels"], val_out["preds"])

        # Siapkan dictionary log dasar
        log_dict = {
            "fold": cur_fold,
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "train_accuracy": train_out["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_out["loss"],
            "val_accuracy": val_out["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }

        # Logika Best Model & Confusion Matrix
        if val_out["loss"] < best_val_loss:
            best_val_loss = val_out["loss"]
            best_epoch = epoch
            best_val_preds = val_out["preds"]
            best_val_labels = val_out["labels"]

            # Tambahkan Confusion Matrix ke dictionary log yang SAMA
            if args.use_wandb or IS_NOTEBOOK:
                log_dict["val_confusion_matrix"] = wandb.plot.confusion_matrix(
                    y_true=val_out["labels"],
                    preds=val_out["preds"],
                    class_names=["non_cyberbullying", "cyberbullying"]
                )
                print(f"Logged confusion matrix for fold {cur_fold} at epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_out['loss']:.4f} | Acc: {train_out['accuracy']:.2f}%")
        print(f"Val   Loss: {val_out['loss']:.4f} | Acc: {val_out['accuracy']:.2f}%")

        # EKSEKUSI LOG WANDB (Hanya 1 kali per epoch)
        if args.use_wandb or IS_NOTEBOOK:
            wandb.log(log_dict)

        # scheduler.step()

        # Early Stopping check
        if early_stopping(val_out["loss"]):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if args.output_model:
            model_save_path = os.path.join(output_dir, f"fold_{cur_fold+1}_model.pth")
            torch.save(model.state_dict(), model_save_path)

    # save_conf_matrix(best_val_labels, best_val_preds)

    return model

def get_device():
    """Get the device to use (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def to_device(data, device):
    """Move data to specified device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def get_gpu_memory():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            "allocated": f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
            "cached": f"{torch.cuda.memory_reserved()/1e9:.2f} GB"
        }
    return None


# In[ ]:


def main():
    args = parse_args()
    set_seed(args.seed)

    model_names = ['default-textcnn'] # ['light-textcnn', 'medium-textcnn', 'weight-textcnn', 'sedepthwise-textcnn']

    for name in model_names:
        global model_name
        model_name = name
    #        -> Here...

        if args.output_model:
            out_dir = create_output_dir(args.output_dir)

        for fold in range(0, args.n_folds):
        # for fold in range(3, 4): # Epoch 3 only
            if args.use_wandb or IS_NOTEBOOK:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wandb.init(
                    project="sentiment-analys-cyberbullying",
                    # group=f"model_{model_name}", #
                    group=args.wandb_group, #
                    # name=f"fold_{fold}_exp_{timestamp}", #
                    name=f"fold_{fold}_exp_{timestamp}", #
                    config=vars(args),
                    notes=args.wandb_note #
                )

            cnn_train(args, output_dir=out_dir, cur_fold=fold, model_name=model_name)

            if args.use_wandb or IS_NOTEBOOK:
                wandb.finish()

if __name__ == "__main__":
    main()

