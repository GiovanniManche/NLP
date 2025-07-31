import torch
from torch import nn
from transformers import BertModel, BertConfig
from config import Config
from torch import Tensor
from typing import Optional, Tuple, Union

class TransformerClassifier(nn.Module) :
   """
   Class that inherits from the neural network module from torch, then proceeds to create
   a binary classification model based on a pre-trained BERT encoder.
   The final hidden state of the [CLS] token is passed through a linear layer.

   Inputs :
   ---------
   pretrained_model_name : tells the code which version of the pretrained model to load 
                          (model's weights are computed beforehand on a large dataset)
   dropout_prob : a random subset of neurons is dropped out 
                 at each training set to avoid overfitting by making the network 
                 less reliant to specific neurons.
   """
   def __init__(self, 
                pretrained_model_name : str = Config.MODEL_NAME, 
                dropout_prob : float = Config.DROPOUT) :
       super().__init__()
       # Load pre-trained BERT
       self.bert = BertModel.from_pretrained(pretrained_model_name)
       hidden_size = self.bert.config.hidden_size

       # Dropout + classification layer
       self.dropout = nn.Dropout(dropout_prob)
       self.classifier = nn.Linear(hidden_size, 2)  # 2 classes (0 and 1)

   def forward(self, 
               input_ids : Tensor, 
               attention_mask : Tensor, 
               token_type_ids : Optional[Tensor] = None, 
               labels : Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]] :
       """
       Forward pass of the model.

       Inputs :
       ----------
       input_ids : Tensor of token IDs
       attention_mask : Tensor indicating non-padded tokens
       token_type_ids : Segment IDs (optional, can be None)
       labels : Optional labels for training

       Returns :
       ----------
       If labels is provided : a tuple (loss, logits)
       Else : logits only
       """
       outputs = self.bert(
           input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids
       )
       cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
       cls_output = self.dropout(cls_output)
       logits = self.classifier(cls_output)

       if labels is not None :
           loss_fn = nn.CrossEntropyLoss()
           loss = loss_fn(logits, labels)
           return loss, logits

       return logits