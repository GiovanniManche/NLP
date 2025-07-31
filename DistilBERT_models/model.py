import torch
from torch import nn
from transformers import DistilBertModel
from config import Config
from torch import Tensor
from typing import Optional, Tuple, Union

class RegularizedDistilBERT(nn.Module) :
   """
   Regularized DistilBERT model for binary classification with aggressive regularization techniques.
   
   This model implements several regularization strategies :
   - Layer freezing : Only the last 2 transformer layers are trainable
   - High dropout rates : Multiple dropout layers with different rates
   - Label smoothing : Reduces overconfidence in predictions
   - Limited fine-tuning : Prevents overfitting on small datasets
   """
   
   def __init__(self) :
       super().__init__()
       
       # Load pre-trained DistilBERT model (smaller and faster than BERT)
       self.distilbert = DistilBertModel.from_pretrained(Config.MODEL_NAME)

       # Freeze embeddings to prevent overfitting on small vocabulary differences
       for param in self.distilbert.embeddings.parameters() :
           param.requires_grad = False
           
       # Freeze all transformer layers except the last 2 for limited fine-tuning
       for layer in self.distilbert.transformer.layer[:-2] :
           for param in layer.parameters() :
               param.requires_grad = False

       # Build classification head with heavy regularization
       self.classifier = nn.Sequential(
           # First dropout layer with high rate to prevent overfitting
           nn.Dropout(Config.DROPOUT1),  # 0.6 dropout rate
           
           # Intermediate linear layer to reduce dimensionality
           nn.Linear(768, Config.INTERMEDIATE_DIM),  # 768 -> 128 dimensions
           
           # ReLU activation for non-linearity
           nn.ReLU(),
           
           # Second dropout layer before final classification
           nn.Dropout(Config.DROPOUT2),  # 0.4 dropout rate
           
           # Final classification layer for binary classification
           nn.Linear(Config.INTERMEDIATE_DIM, 2)  # 128 -> 2 classes (Human vs AI)
       )

   def forward(self, 
               input_ids : Tensor, 
               attention_mask : Tensor, 
               labels : Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]] :
       """
       Forward pass of the regularized DistilBERT model.

       Inputs :
       ----------
       input_ids : Tensor of token IDs from tokenizer
       attention_mask : Tensor indicating which tokens are real vs padding
       labels : Optional labels for training (None during inference)

       Returns :
       ----------
       If labels provided : tuple of (loss, logits) for training
       If no labels : logits only for inference
       """
       # Pass input through DistilBERT encoder
       outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
       
       # Extract [CLS] token representation (first token contains sentence-level info)
       cls_output = outputs.last_hidden_state[:, 0]  # Shape: [batch_size, 768]
       
       # Pass through regularized classification head
       logits = self.classifier(cls_output)  # Shape: [batch_size, 2]

       # Calculate loss during training phase
       if labels is not None :
           # Use CrossEntropyLoss with label smoothing to reduce overconfidence
           loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
           loss = loss_fn(logits, labels)
           return loss, logits

       # Return only logits during inference
       return logits