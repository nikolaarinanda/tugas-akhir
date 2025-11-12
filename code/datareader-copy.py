import random
import re
import os
import json
import nltk
import torch
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

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

# STOPWORDS = set(stopwords.words('indonesian'))
STOPWORDS = set(stopwords.words('english'))

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
        # self.vocab_size = len(self.tokenizer)

        self.load_data()
        self.setup_folds()
        self.setup_indices()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        text, label = map(str, self.df.iloc[idx][["comment", "sentiment"]])
        comment_processed = self.preprocess(text)
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

    # Augmentation functions

    def random_typo(self, text):
        words = text.split()
        if len(words) < 1:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_list = list(word)
            i = random.randint(0, len(char_list) - 2)
            char_list[i], char_list[i+1] = char_list[i+1], char_list[i]
            words[idx] = ''.join(char_list)
        return ' '.join(words)

    def random_delete(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        return ' '.join(words)

    def augmentation_text(self, text, prob=0.5):
        if random.random() > prob:
            return text
        # Random typo
        if random.random() < 0.5:
            print("Applying random typo augmentation")
            text = self.random_typo(text)
        # Random delete
        if random.random() < 0.5:
            print("Applying random delete augmentation")
            text = self.random_delete(text)
        return text

    def preprocess(self, text):
        '''
        Pre-processing teks seperti menghapus selain alphanumeric, lowercasing, augmentasi, tokenisasi dan stopword removal
        '''
        text = re.sub(r'[^\w\s]|[^\x00-\x7F]', '', text)
        text = text.lower()
        text = self.augmentation_text(text, prob=0.5)
        words = nltk.word_tokenize(text)
        # lemmatizer = WordNetLemmatizer() # Stemming atau Lemmatization (gunakan salah satu)
        # words = [lemmatizer.lemmatize(word) for word in words]
        words = [word for word in words if word not in STOPWORDS]
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
        '''
        Mempersiapkan n-Fold Cross Validation
        '''
        if os.path.exists(self.folds_file):
            self.load_folds()
        else:
            self.create_folds()

    def load_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
        self.fold_indices = fold_data['fold_indices']
    
    def create_folds(self):
        '''
        Apabila fold sudah ada, maka load fold
        '''
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
            print(f'Created {self.n_folds}-fold indices and saved to {self.folds_file}')

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df.columns = ['sentiment', 'comment']
        self.df = self.df.dropna(subset=['sentiment', 'comment'])
        self.df['sentiment'] = self.df['sentiment'].astype(int)
        self.df['sentiment'] = self.df['sentiment'].apply(lambda x: 1 if x == -1 else 0)
        self.df = self.df[(self.df['sentiment'] == 0) | (self.df['sentiment'] == 1)]

        # if self.split == "train": # Undersampling (hanya untuk split "train")
            # df_label_0 = self.df[self.df['sentiment'] == 0]
            # df_label_1 = self.df[self.df['sentiment'] == 1]

            # min_samples_per_class = min(len(df_label_0), len(df_label_1))

            # df_label_0_undersampled = df_label_0.sample(n=min_samples_per_class, random_state=self.random_state)
            # df_label_1_undersampled = df_label_1.sample(n=min_samples_per_class, random_state=self.random_state)

            # self.df = pd.concat([df_label_0_undersampled, df_label_1_undersampled])
            # self.df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

if __name__ == "__main__":
    # dataset = CyberbullyingDataset(fold=0, split="train")
    # data = dataset[0]
    # print("Label:", data['labels'].item())
    # print("Original Text:", data['original_text'])
    # print("Processed Text:", data['processed_text'])

    dataset = CyberbullyingDataset(fold=random.randint(0, 4), split=random.choice(["train", "val"])) # Instansiasi kelas dengan fold dan split acak
    data = dataset[random.randint(0, len(dataset) - 1)] # Ambil data secara acak
    print(f"original text: {data['original_text']}")
    print(f"processed text: {data['processed_text']}")