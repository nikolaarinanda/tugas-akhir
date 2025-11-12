import torch
from torch import nn 
# from torch import torch.nn as nn
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