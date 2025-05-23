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
