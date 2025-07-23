import torch
from torch import nn # = import torch.nn as nn
import torch.nn.functional as F

class TextCNNLight(nn.Module):
    """
    TextCNNLight is a lightweight version of TextCNN for text classification.
    vocab_size: Size of the vocabulary
    embed_dim: Dimension of the word embeddings
    num_classes: Number of output classes
    output_dim: Dimension of the output after convolution
    kernel_size: List of kernel sizes for convolutional layers
    This model uses two convolutional layers with kernel sizes 3 and 4,panjang jendela (window) digunakan filter untuk melihat urutan kata secara lokal.
    followed by max pooling and a fully connected layer.
    The output dimension is set to 64, and dropout is applied to prevent overfitting.
    The model is designed to be efficient and suitable for smaller datasets or real-time applications.
    """
    def __init__(self, vocab_size, embed_dim=100, num_classes=2, output_dim=64, dropout_rate=0.3):
        super(TextCNNLight, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, output_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, output_dim, kernel_size=4)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear((output_dim * 2), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNMedium(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNNMedium, self).__init__()
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

class TextCNNHeavy(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        num_classes=2,
        output_channels=256,
        kernel_sizes=[3, 4, 5],
        dropout_rate=0.5
    ):
        super(TextCNNHeavy, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, output_channels, kernel_size=K),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
            for K in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        
        fc_input_dim = output_channels * len(kernel_sizes) 
        
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        x = self.embedding(x)
        # Ubah dimensi untuk Conv1d: [batch_size, embed_dim, sequence_length]
        x = x.permute(0, 2, 1) 
        
        pooled_outputs = []
        for conv_block in self.convs:
            # max-pooling di dimensi sequence_length
            pooled = conv_block(x).max(dim=2)[0]
            pooled_outputs.append(pooled)
        
        # Gabungkan semua output pooling
        x = torch.cat(pooled_outputs, dim=1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
