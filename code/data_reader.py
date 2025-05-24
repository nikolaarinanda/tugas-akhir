import pandas as pd
import numpy as np
import random
import re
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer

# Download resource NLTK yang diperlukan
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

INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))

class CyberbullyingDataset(Dataset):
    def __init__(
            self,
            file_path="../dataset/Dataset-Research.csv",
            tokenizer_name="indobenchmark/indobert-base-p1",
            folds_file="cyberbullying_datareader_simple_folds.json",
            random_state=29082002,
            split="train",
            fold=0,
            n_folds=5,
            max_length=128,
            augmentasi_file="../dataset/dictionary/augmentation.json",
            slang_word_file="../dataset/dictionary/slang-word-specific.json"
    ):        
        self.file_path = file_path
        self.folds_file = folds_file
        self.random_state = random_state
        self.split = split
        self.fold = fold
        self.n_folds = n_folds
        self.max_length = max_length
        self.augmentasi_data = self.load_file(augmentasi_file)
        self.slang_dict = self.load_file(slang_word_file)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Load dataset
        self.load_data()
        # Setup n-Fold Cross Validation => membuat self.fold_indices yang berisi train_indices dan val_indices dari semua fold
        self.setup_folds()
        # Mempersiapkan Indices (bentuk jamak index) => membuat self.indices (yang berisi indices dari fold yang dipilih berdasarkan 'split' dan 'fold'
        self.setup_indices()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Mengambil index dari data yang akan diambil
        idx = self.indices[idx]
        # Mengambil data komentar dan sentiment
        comment = str(self.df.iloc[idx]['comment'])
        sentiment = str(self.df.iloc[idx]['sentiment'])
        # Melakukan Pre-Processing
        comment_processed = self.preprocess(comment)
        # Tokenisasi
        encoding = self.tokenizer(
            comment_processed,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        data = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(sentiment), dtype=torch.long),
            'original_text': comment,
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

    def random_typo(self, text):
        words = text.split()
        if len(words) < 1:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_list = list(word)
            i = random.randint(0, len(char_list) - 2)
            char_list[i], char_list[i+1] = char_list[i+1], char_list[i]  # swap 2 huruf berdekatan
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
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        return ' '.join(words)

    def augmentation_text(self, text, p_phrase=1.0, p_synonym=1.0, p_typo=1.0, p_swap=1.0, p_random_delete=1.0):
        # Augmentasi frasa
        if random.random() < p_phrase:
            for phrase, replacements in self.augmentasi_data.get("replace_phrases", {}).items():
                if phrase in text:
                    text = text.replace(phrase, random.choice(replacements))
        # Augmentasi sinonim
        if random.random() < p_synonym:
            words = text.split()
            for i, word in enumerate(words):
                if word in self.augmentasi_data.get("synonyms", {}):
                    words[i] = random.choice(self.augmentasi_data["synonyms"][word])
            text = ' '.join(words)    
        # Random typo
        if random.random() < p_typo:
            text = self.random_typo(text)
        # Random swap
        if random.random() < p_swap:
            text = self.random_swap(text)
        # Random delete
        if random.random() < p_random_delete:
            text = self.random_delete(text)
        return text

    def normalization(self, words):        
        # Normalisasi setiap kata
        normalized_words = []
        for word in words:
            # Cek apakah kata ada di dictionary slang
            if word in self.slang_dict:
                normalized_words.append(self.slang_dict[word])
            else:
                normalized_words.append(word)
        
        # Gabungkan kembali menjadi text
        return normalized_words

    def preprocess(self, text):
        # Konversi ke huruf kecil
        text = text.lower()

        # Hapus URL
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Hapus mention (@...) dan hashtag (#...)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Menghapus special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Hapus emoji dan karakter non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Augmentasi
        text = self.augmentation_text(text, p_phrase=0.5, p_synonym=0.5, p_typo=0.5, p_swap=0.5, p_random_delete=0.5)
        
        # Tokenisasi
        words = nltk.word_tokenize(text)

        # Normalisasi
        words = self.normalization(words)

        # Menghapus stopwords
        words = [word for word in words if word not in INDONESIAN_STOPWORDS]

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
        else:
            # Jika tidak ada, maka buat fold
            self.create_folds()

    def load_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
        self.fold_indices = fold_data['fold_indices']
        # print(f"Menggunakan {fold_data['n_folds']} folds dengan {fold_data['n_samples']} samples")
    
    def create_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
        print(f"Membuat n-fold CV dengan random state {self.random_state}")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
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
            # print(f'Created {self.n_folds}-fold indices and saved to {self.folds_file}')

    def load_data(self):
        self.df = pd.read_csv(self.file_path) # Load csv
        self.df.columns = ['sentiment', 'comment'] # Rename columns
        self.df = self.df.dropna(subset=['sentiment', 'comment']) # Drop NaN values
        self.df['sentiment'] = self.df['sentiment'].astype(int) # Convert sentiment to int
        self.df['sentiment'] = self.df['sentiment'].apply(lambda x: 1 if x == -1 else 0) # Transform labels: convert -1 to 1, and 1 to 0
        self.df = self.df[(self.df['sentiment'] == 0) | (self.df['sentiment'] == 1)] # Filter sentiment
    
if __name__ == "__main__":
    # dataset = CyberbullyingDataset(fold=random.randint(0, 4), split=random.choice(["train", "val"])) # Instansiasi kelas dengan fold dan split acak
    # data = dataset[random.randint(0, len(dataset) - 1)] # Ambil data secara acak

    dataset = CyberbullyingDataset(fold=0, split="train") # instansiasi kelas

    for data in dataset:
        print(f"original text: {data['original_text']}")
        print(f"processed text: {data['processed_text']}")

    # data = dataset[8] # mengambil data

    # print(f"input IDs: {data['input_ids']}")
    # print(f"attention mask: {data['attention_mask']}")
    # print(f"original text: {data['original_text']}")
    # print(f"processed text: {data['processed_text']}")
    # print(f"original index: {data['original_index']}")
    # print(f"labels: {data['labels']}")

    # print(data)

    