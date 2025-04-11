import pandas as pd
import numpy as np
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
import random
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer

# Download resource NLTK yang diperlukan
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Mencoba download stopwords bahasa Indonesia jika tersedia, jika tidak gunakan bahasa Inggris
try:
    nltk.data.find('corpora/stopwords/indonesian')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        print("Peringatan: Tidak dapat mengunduh stopwords bahasa Indonesia, menggunakan stopwords bahasa Inggris sebagai gantinya")

# Mendefinisikan stopwords bahasa Indonesia (daftar singkat) jika NLTK tidak memilikinya
INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam', 'akan',
    'pada', 'juga', 'saya', 'ke', 'karena', 'tersebut', 'bisa', 'ada', 'itu', 'atau',
    'seperti', 'oleh', 'menjadi', 'adalah', 'ya', 'nya', 'kalo', 'yg', 'dgn', 'gak', 
    'tdk', 'gk', 'aja', 'sih', 'udah', 'sudah', 'nggak', 'ngga', 'nih', 'tau', 'tahu'
}

class ShopeeCommentDataReader:
    """
    Pembaca data untuk dataset komentar Shopee dengan validasi silang 5-fold
    """
    def __init__(self, 
                 file_path: str,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 128,
                 n_folds: int = 5,
                 random_state: int = 42,
                 batch_size: int = 16,
                 apply_preprocessing: bool = True,
                 apply_augmentation: bool = False,
                 augmentation_prob: float = 0.3,
                 folds_file: str = "shopee_folds.json",
                 language: str = "indonesian"):
        """
        Inisialisasi pembaca data dengan validasi silang k-fold
        
        Args:
            file_path: Path ke file Excel
            tokenizer_name: Nama tokenizer pre-trained yang digunakan
            max_length: Panjang maksimum sequence untuk tokenization
            n_folds: Jumlah fold untuk validasi silang
            random_state: Random seed untuk reproduksibilitas
            batch_size: Ukuran batch untuk DataLoader
            apply_preprocessing: Apakah akan menerapkan preprocessing teks
            apply_augmentation: Apakah akan menerapkan augmentasi teks
            augmentation_prob: Probabilitas menerapkan augmentasi pada setiap sampel
            folds_file: Path untuk menyimpan/memuat indeks fold
            language: Bahasa data teks ('indonesian' atau 'english')
        """
        self.file_path = file_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.n_folds = n_folds
        self.random_state = random_state
        self.batch_size = batch_size
        self.apply_preprocessing = apply_preprocessing
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.folds_file = folds_file
        self.language = language.lower()
        
        # Import tokenizer di sini untuk menghindari masalah saat memuat modul
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Inisialisasi lemmatizer untuk teks bahasa Inggris
        self.lemmatizer = WordNetLemmatizer()
        
        # Siapkan stopwords berdasarkan bahasa
        if self.language == 'indonesian':
            try:
                # Coba gunakan stopwords bahasa Indonesia NLTK jika tersedia
                self.stop_words = set(stopwords.words('indonesian'))
                print(f"Stopwords bahasa Indonesia ditemukan di NLTK")
            except:
                # Fallback ke daftar yang telah ditentukan sebelumnya
                self.stop_words = INDONESIAN_STOPWORDS
                print(f"Stopwords bahasa Indonesia tidak ditemukan di NLTK, menggunakan daftar yang telah ditentukan")
        else:
            # Default ke stopwords bahasa Inggris
            self.stop_words = set(stopwords.words('english'))
        
        # Akan diinisialisasi dalam load_data
        self.df = None
        self.text_column = None
        self.fold_indices = None
        self.current_fold = 0
        self.class_weights = None
    
    def _create_dataset(self, comments: List[str], ratings: List[int], apply_augmentation: bool = False):
        """
        Buat dataset dari komentar dan rating
        
        Args:
            comments: Daftar teks komentar
            ratings: Daftar rating (1-5)
            apply_augmentation: Apakah akan menerapkan augmentasi teks
            
        Returns:
            Dataset untuk digunakan dengan DataLoader
        """
        
        class _Dataset(Dataset):
            def __init__(self, comments, ratings, tokenizer, max_length, apply_augmentation, augmentation_prob, parent):
                self.comments = comments
                self.ratings = ratings
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.apply_augmentation = apply_augmentation
                self.augmentation_prob = augmentation_prob
                self.parent = parent
                
            def __len__(self):
                return len(self.comments)
            
            def __getitem__(self, idx):
                comment = str(self.comments[idx])
                rating = self.ratings[idx]
                
                # Terapkan augmentasi jika diaktifkan dan ambang probabilitas terpenuhi
                if self.apply_augmentation and random.random() < self.augmentation_prob:
                    comment = self.parent._augment_text(comment)
                
                encoding = self.tokenizer(
                    comment,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(rating - 1, dtype=torch.long)  # Konversi ke rentang 0-4
                }
        
        return _Dataset(comments, ratings, self.tokenizer, self.max_length, 
                       apply_augmentation, self.augmentation_prob, self)
    
    def _augment_text(self, text: str) -> str:
        """
        Menerapkan teknik augmentasi teks
        
        Args:
            text: Teks asli
        
        Returns:
            Teks yang telah diaugmentasi
        """
        augmentation_type = random.choice(['synonym_replacement', 'random_deletion', 
                                         'random_swap', 'backtranslation'])
        
        if augmentation_type == 'synonym_replacement':
            # Simulasi penggantian kata sederhana untuk demonstrasi
            # Dalam produksi, gunakan penggantian sinonim yang tepat dengan WordNet
            words = text.split()
            if len(words) > 3:
                idx_to_replace = random.randint(0, len(words) - 1)
                common_replacements = {
                    'good': ['great', 'excellent', 'nice', 'wonderful'],
                    'bad': ['poor', 'terrible', 'awful', 'horrible'],
                    'product': ['item', 'merchandise', 'good', 'purchase'],
                    'like': ['enjoy', 'appreciate', 'love', 'adore'],
                    'delivery': ['shipping', 'shipment', 'arrival', 'transport']
                }
                
                if words[idx_to_replace].lower() in common_replacements:
                    words[idx_to_replace] = random.choice(common_replacements[words[idx_to_replace].lower()])
                
                return ' '.join(words)
            
        elif augmentation_type == 'random_deletion':
            words = text.split()
            if len(words) > 5:  # Hanya hapus jika kita memiliki cukup kata
                delete_prob = 0.1  # Probabilitas untuk menghapus setiap kata
                new_words = [word for word in words if random.random() > delete_prob]
                
                # Memastikan kita tidak menghapus semua kata
                if not new_words:
                    new_words = words
                
                return ' '.join(new_words)
            
        elif augmentation_type == 'random_swap':
            words = text.split()
            if len(words) > 3:
                num_swaps = max(1, int(len(words) * 0.1))
                for _ in range(num_swaps):
                    idx1, idx2 = random.sample(range(len(words)), 2)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                
                return ' '.join(words)
        
        elif augmentation_type == 'backtranslation':
            # Dalam implementasi nyata, gunakan API terjemahan
            # Di sini kita hanya akan mensimulasikannya dengan perubahan kecil
            words = text.split()
            for i in range(len(words)):
                if random.random() < 0.15:  # 15% kemungkinan untuk memodifikasi setiap kata
                    if len(words[i]) > 3:
                        # Acak huruf tengah, simpan huruf pertama dan terakhir
                        middle = list(words[i][1:-1])
                        random.shuffle(middle)
                        words[i] = words[i][0] + ''.join(middle) + words[i][-1]
            
            return ' '.join(words)
        
        # Jika tidak ada augmentasi yang berhasil atau berlaku, kembalikan teks asli
        return text
        
    def load_data(self) -> None:
        """
        Memuat dan memproses data dari file Excel
        Mengatur fold validasi silang
        """
        # Baca file Excel
        self.df = pd.read_excel(self.file_path)
        
        # Nama kolom yang diperbarui berdasarkan struktur yang benar
        # Periksa jumlah kolom dan tetapkan kolom dengan benar
        if len(self.df.columns) == 4:
            self.df.columns = ['userName', 'rating', 'timestamp', 'comment']
        else:
            # Jika jumlah kolom tidak sesuai dengan yang diharapkan, cetak kolom sebenarnya
            print(f"Peringatan: Diharapkan 4 kolom tetapi ditemukan {len(self.df.columns)}")
            print(f"Kolom aktual: {self.df.columns.tolist()}")
            # Coba identifikasi kolom yang benar berdasarkan tipe data
            for i, col in enumerate(self.df.columns):
                sample_value = self.df[col].iloc[0] if not self.df.empty else None
                print(f"Kolom {i}: '{col}', Contoh nilai: '{sample_value}', Tipe: {type(sample_value)}")
        
        # Pembersihan data dasar
        self.df = self.df.dropna(subset=['comment', 'rating'])
        self.df['rating'] = self.df['rating'].astype(int)
        
        # Validasi rating berada dalam rentang 1-5
        self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)]
        
        # Proses komentar jika diaktifkan
        if self.apply_preprocessing:
            self.df['processed_comment'] = self.df['comment'].apply(self._preprocess_text)
            self.text_column = 'processed_comment'
        else:
            self.text_column = 'comment'
        
        # Hitung bobot kelas untuk menangani ketidakseimbangan
        class_counts = self.df['rating'].value_counts().sort_index()
        self.class_weights = torch.FloatTensor(
            [len(self.df) / (len(class_counts) * count) for count in class_counts]
        )
        
        # Siapkan fold validasi silang
        self._setup_folds()
        
        print(f"Data dimuat dan diproses:")
        print(f"  - Total sampel: {len(self.df)}")
        print(f"  - Distribusi kelas: {class_counts.to_dict()}")
        print(f"  - Jumlah fold: {self.n_folds}")
    
    def _setup_folds(self) -> None:
        """
        Siapkan fold validasi silang atau muat dari file jika ada
        """
        # Periksa apakah definisi fold sudah ada
        if os.path.exists(self.folds_file):
            self._load_folds()
        else:
            self._create_folds()
            
    def _create_folds(self) -> None:
        """
        Buat indeks k-fold stratifikasi dan simpan ke file
        """
        # Inisialisasi stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Hasilkan indeks train/val untuk validasi silang
        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['rating'])):
            fold_indices[f"fold_{fold}"] = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            }
        
        # Simpan ke file
        fold_data = {
            'fold_indices': fold_indices,
            'created_at': pd.Timestamp.now().isoformat(),
            'n_samples': len(self.df),
            'n_folds': self.n_folds,
            'random_state': self.random_state
        }
        
        with open(self.folds_file, 'w') as f:
            json.dump(fold_data, f)
            
        self.fold_indices = fold_indices
        
    def _load_folds(self) -> None:
        """
        Muat indeks fold dari file
        """
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
            
        self.fold_indices = fold_data['fold_indices']
        
        # Verifikasi bahwa fold yang dimuat kompatibel dengan data saat ini
        if len(self.df) != fold_data['n_samples']:
            print(f"Peringatan: Ukuran dataset ({len(self.df)}) berbeda dari saat fold dibuat ({fold_data['n_samples']})")
            print("Membuat ulang fold...")
            os.remove(self.folds_file)
            self._create_folds()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Memproses teks dengan menghapus karakter khusus, 
        stopwords, dan lemmatizing (untuk bahasa Inggris) atau hanya menghapus stopwords (untuk bahasa Indonesia)
        
        Args:
            text: Teks untuk diproses
            
        Returns:
            Teks yang telah diproses
        """
        if not isinstance(text, str):
            return ""
        
        # Konversi ke huruf kecil
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Hapus emoji dan karakter khusus, tapi pertahankan karakter bahasa Indonesia
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenisasi sederhana dengan spasi untuk menghindari masalah NLTK punkt
        tokens = text.split()
        
        # Hapus stopwords
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        
        # Terapkan lemmatization hanya untuk teks bahasa Inggris
        if self.language == 'english':
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Gabungkan kembali token
        return ' '.join(tokens)
    
    def get_fold(self, fold: int) -> Tuple[DataLoader, DataLoader]:
        """
        Dapatkan train dan validation DataLoader untuk fold tertentu
        
        Args:
            fold: Nomor fold (0 sampai n_folds-1)
            
        Returns:
            Tuple dari (train_dataloader, val_dataloader) untuk fold yang ditentukan
        """
        if self.df is None:
            self.load_data()
            
        if fold < 0 or fold >= self.n_folds:
            raise ValueError(f"Nomor fold harus antara 0 dan {self.n_folds-1}")
            
        self.current_fold = fold
        
        # Dapatkan indeks khusus fold
        fold_key = f"fold_{fold}"
        train_indices = self.fold_indices[fold_key]['train_indices']
        val_indices = self.fold_indices[fold_key]['val_indices']
        
        # Buat dataset
        train_dataset = self._create_dataset(
            comments=self.df.iloc[train_indices][self.text_column].values,
            ratings=self.df.iloc[train_indices]['rating'].values,
            apply_augmentation=self.apply_augmentation
        )
        
        val_dataset = self._create_dataset(
            comments=self.df.iloc[val_indices][self.text_column].values,
            ratings=self.df.iloc[val_indices]['rating'].values,
            apply_augmentation=False  # Tidak ada augmentasi untuk set validasi
        )
        
        # Buat dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        print(f"Fold {fold+1} dimuat:")
        print(f"  - Sampel training: {len(train_dataset)}")
        print(f"  - Sampel validasi: {len(val_dataset)}")
        
        return train_dataloader, val_dataloader
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Dapatkan bobot kelas untuk menangani ketidakseimbangan kelas
        
        Returns:
            Tensor dari bobot kelas
        """
        if self.class_weights is None:
            self.load_data()
            
        return self.class_weights
    
    def get_tokenizer(self):
        """
        Dapatkan tokenizer
        
        Returns:
            Objek tokenizer
        """
        return self.tokenizer


# Contoh penggunaan
if __name__ == "__main__":
    data_reader = ShopeeCommentDataReader(
        file_path="../dataset/dataset_shopee.xlsx",
        apply_preprocessing=True,
        apply_augmentation=True,
        batch_size=16
    )
    
    data_reader.load_data()
    
    # Cetak struktur kolom dari dataframe yang dimuat
    print("Info DataFrame:")
    print(data_reader.df.info())
    print("\nBeberapa baris pertama:")
    print(data_reader.df.head())
    
    # Contoh: akses fold 0
    train_loader, val_loader = data_reader.get_fold(0)
    
    # Cetak bobot kelas
    print(f"Bobot kelas: {data_reader.get_class_weights()}")
    
    # Pemeriksaan sampel data
    for batch in train_loader:
        print(f"Ukuran batch: {len(batch['input_ids'])}")
        print(f"Panjang maksimum sequence: {batch['input_ids'].size(1)}")
        print(f"Label: {batch['labels']}")
        
        # Decode satu sampel untuk verifikasi
        sample_idx = 0
        sample_tokens = batch['input_ids'][sample_idx]
        sample_text = data_reader.tokenizer.decode(sample_tokens, skip_special_tokens=True)
        print(f"Contoh teks: {sample_text}")
        print(f"Label: {batch['labels'][sample_idx] + 1} bintang")  # Konversi kembali ke rentang 1-5
        break