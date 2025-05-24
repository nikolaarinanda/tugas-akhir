import torch
from torch import nn # import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERTSentimentClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)  # dua kelas output

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output['pooler_output']
        dropped_output = self.dropout(pooled_output)
        return self.linear(dropped_output)

class CNNSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2, kernel_sizes=[3,4,5], num_filters=100, dropout_rate=0.1):
        super(CNNSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [Batch, Max_seq_len, Embed_dim]
        x = x.unsqueeze(1)  # Add channel dimension: [Batch, Channel=1, Max_seq_len, Embed_dim]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply convolution and ReLU
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Max pooling over time
        x = torch.cat(x, 1)  # Concatenate feature maps from different kernels
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class CombinedTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2, kernel_sizes=[3, 4, 5], num_filters=100, dropout_rate=0.5, padding_idx=0):
        super(CombinedTextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # ModuleList untuk membuat layer konvolusi secara dinamis
        # Setiap Conv1d akan memiliki embed_dim sebagai in_channels dan num_filters sebagai out_channels
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=k) 
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer fully connected: inputnya adalah jumlah total filter dari semua kernel
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: [Batch_size, Max_seq_len]
        
        x = self.embedding(x)  # [Batch_size, Max_seq_len, Embed_dim]
        
        # Permute x untuk sesuai dengan input Conv1d: [Batch_size, Embed_dim, Max_seq_len]
        x = x.permute(0, 2, 1)
        
        # Terapkan konvolusi, ReLU, dan max-pooling untuk setiap kernel
        # conved_outputs akan menjadi list dari tensor, masing-masing [Batch_size, Num_filters]
        conved_outputs = []
        for conv_layer in self.convs:
            conved = conv_layer(x)    # [Batch_size, Num_filters, Seq_len_out]
            activated = F.relu(conved) # [Batch_size, Num_filters, Seq_len_out]
            
            # Max-pooling over time
            # activated.size(2) adalah dimensi panjang sekuens setelah konvolusi
            pooled = F.max_pool1d(activated, activated.size(2)).squeeze(2) # [Batch_size, Num_filters]
            conved_outputs.append(pooled)
            
        # Gabungkan hasil dari semua kernel
        # x_cat: [Batch_size, Num_filters * len(kernel_sizes)]
        x_cat = torch.cat(conved_outputs, dim=1)
        
        x = self.dropout(x_cat)
        logits = self.fc(x)  # [Batch_size, Num_classes]
        
        return logits

class SimpleLightweightTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, out_channels=50, kernel_size=3, dropout_rate=0.1, padding_idx=0):
        """
        Inisialisasi model CNN sederhana.

        Args:
            vocab_size (int): Ukuran kosakata (jumlah kata unik).
            embed_dim (int): Dimensi vektor embedding kata.
            num_classes (int): Jumlah kelas output.
            out_channels (int): Jumlah filter (atau channel output) untuk layer konvolusi.
                                Ini menentukan jumlah fitur yang diekstrak oleh layer konvolusi.
            kernel_size (int): Ukuran jendela konvolusi (misalnya, 3 berarti melihat 3 kata sekaligus).
            dropout_rate (float): Tingkat dropout untuk regularisasi.
            padding_idx (int): Indeks token padding di embedding.
        """
        super(SimpleLightweightTextCNN, self).__init__()
        
        # --- Lapisan-lapisan Model ---

        # 1. Layer Embedding:
        # Mengubah input berupa urutan indeks kata menjadi urutan vektor embedding.
        # Misalnya, jika kata "kucing" memiliki indeks 5, layer ini akan mengubah 5 menjadi vektor [0.1, 0.3, ..., 0.9].
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # 2. Layer Konvolusi 1D (nn.Conv1d):
        # Menerapkan filter (kernel) pada sekuens embedding untuk mengekstrak fitur lokal.
        # Bayangkan filter ini seperti "jendela geser" yang mencari pola tertentu (misal, pola 3 kata).
        # - in_channels: embed_dim (karena setiap dimensi embedding dianggap sebagai "channel" input)
        # - out_channels: jumlah filter yang ingin kita pelajari. Setiap filter akan belajar mendeteksi pola yang berbeda.
        # - kernel_size: lebar jendela filter (berapa banyak kata yang dilihat sekaligus).
        self.conv1 = nn.Conv1d(in_channels=embed_dim, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size)
        
        # 3. Layer Dropout:
        # Untuk mencegah overfitting dengan "mematikan" beberapa neuron secara acak selama training.
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. Layer Fully Connected (Linear):
        # Layer klasifikasi akhir. Mengambil fitur yang telah diproses dan menghasilkan skor untuk setiap kelas.
        # Inputnya adalah 'out_channels' karena setelah max-pooling, kita akan memiliki satu nilai per filter.
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, text_indices):
        """
        Definisi alur kerja forward pass.

        Args:
            text_indices (torch.Tensor): Tensor input berisi indeks kata.
                                         Dimensi: [batch_size, max_seq_len]
        
        Returns:
            torch.Tensor: Logits (skor mentah) untuk setiap kelas.
                          Dimensi: [batch_size, num_classes]
        """
        
        # 1. Lewatkan input melalui Layer Embedding
        # text_indices: [batch_size, max_seq_len]
        # embedded: [batch_size, max_seq_len, embed_dim]
        embedded = self.embedding(text_indices)
        
        # Untuk nn.Conv1d, input diharapkan dalam format: [batch_size, channels, seq_len]
        # Di sini, 'channels' adalah embed_dim, dan 'seq_len' adalah max_seq_len.
        # Jadi, kita perlu menukar (permute) dua dimensi terakhir.
        # embedded (permuted): [batch_size, embed_dim, max_seq_len]
        embedded_permuted = embedded.permute(0, 2, 1)
        
        # 2. Lewatkan hasil embedding melalui Layer Konvolusi
        # conved: [batch_size, out_channels, (max_seq_len - kernel_size + 1)]
        # Ukuran dimensi sekuens akan berkurang sedikit setelah konvolusi (tergantung kernel_size dan padding).
        conved = self.conv1(embedded_permuted)
        
        # 3. Terapkan Fungsi Aktivasi (ReLU direkomendasikan)
        # ReLU (Rectified Linear Unit) mengenalkan non-linearitas.
        # activated: [batch_size, out_channels, (max_seq_len - kernel_size + 1)] (nilai < 0 menjadi 0)
        activated = F.relu(conved)
        
        # 4. Terapkan Max-Pooling (Max-over-time Pooling)
        # Mengambil nilai maksimum dari setiap filter di seluruh dimensi sekuens.
        # Ini mengekstrak fitur yang paling "penting" atau "aktif" yang dideteksi oleh setiap filter.
        # activated.size(2) adalah panjang sekuens setelah konvolusi.
        # pooled: [batch_size, out_channels, 1] (setelah max_pool1d)
        # pooled_squeezed: [batch_size, out_channels] (setelah squeeze)
        pooled = F.max_pool1d(activated, activated.size(2))
        pooled_squeezed = pooled.squeeze(2) 
        
        # 5. Lewatkan hasil pooling melalui Layer Dropout
        # dropped_out: [batch_size, out_channels]
        dropped_out = self.dropout(pooled_squeezed)
        
        # 6. Lewatkan hasil dropout melalui Layer Fully Connected
        # logits: [batch_size, num_classes]
        logits = self.fc(dropped_out)
        
        return logits