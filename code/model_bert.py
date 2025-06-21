import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig

class BertSentimentClassifier(nn.Module):
    """
    Sentiment Classification Model based on pre-trained BERT.
    This model utilizes AutoModelForSequenceClassification from Hugging Face
    which already includes a classification head on top of the BERT encoder.
    """
    def __init__(self, pretrained_model_name: str, num_classes: int):
        """
        Initializes the BertSentimentClassifier.

        Args:
            pretrained_model_name (str): The name of the pre-trained BERT model
                                         (e.g., 'indobenchmark/indobert-base-p1').
            num_classes (int): The number of output classes for classification.
        """
        super(BertSentimentClassifier, self).__init__()
        print(f"Initializing BertSentimentClassifier with pretrained model: {pretrained_model_name}")
        # Load the pre-trained BERT model with a classification head
        # AutoModelForSequenceClassification automatically configures the final layer
        # based on `num_labels`.
        try:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes # Number of labels for the classification head
            )
        except Exception as e:
            print(f"Error loading pretrained BERT model: {e}")
            print("Attempting to load config first and then model.")
            config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)
            self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BERT sentiment classifier.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs (tokenized sentences).
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.
                                           Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Logits (raw scores) for each class.
                          Shape: (batch_size, num_classes)
        """
        # The forward pass of AutoModelForSequenceClassification returns an object
        # with different attributes, including 'logits'.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# class BertSentimentClassifier(torch.nn.Module):
#     """Model Klasifikasi Sentimen berbasis BERT."""
#     def __init__(self, pretrained_model_name, num_classes):
#         super(BertSentimentClassifier, self).__init__()
#         # AutoModelForSequenceClassification sudah dilengkapi dengan classification head
#         self.bert = AutoModelForSequenceClassification.from_pretrained(
#             pretrained_model_name, 
#             num_labels=num_classes
#         )

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         # `logits` adalah output langsung dari classification head AutoModelForSequenceClassification
#         return outputs.logits