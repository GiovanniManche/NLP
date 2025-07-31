import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import Module
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from model import RegularizedDistilBERT
from config import Config
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
from typing import Tuple, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import BERTDataset

def train_and_evaluate() -> None :
   """
   Complete training and evaluation pipeline for DistilBERT classifier.
   
   This function handles the entire machine learning workflow :
   1. Data loading and preprocessing
   2. Model initialization and training with early stopping
   3. Final evaluation on test set with detailed metrics
   """
   # Load dataset with proper encoding for economic texts
   df = pd.read_csv(Config.CSV_PATH, sep=";", encoding="ISO-8859-1")
   
   # Split dataset : 70% train, 21% validation, 30% test (stratified by label)
   train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
   train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)

   # Initialize DistilBERT tokenizer for text preprocessing
   from transformers import DistilBertTokenizer
   tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)

   # Create datasets for each split using custom BERTDataset class
   train_dataset = BERTDataset(train_df, tokenizer, max_length=Config.MAX_LENGTH)
   val_dataset = BERTDataset(val_df, tokenizer, max_length=Config.MAX_LENGTH)
   test_dataset = BERTDataset(test_df, tokenizer, max_length=Config.MAX_LENGTH)

   # Create data loaders for batch processing during training
   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Small batch for memory efficiency
   val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
   test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

   # Initialize regularized DistilBERT model and move to computing device
   model = RegularizedDistilBERT().to(Config.DEVICE)
   
   # Setup AdamW optimizer with weight decay for regularization
   optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

   # Early stopping parameters to prevent overfitting
   best_val_acc : float = 0
   patience : int = 0
   max_patience : int = 3

   # Training loop over multiple epochs
   for epoch in range(Config.EPOCHS) :
       # Set model to training mode (enables dropout and batch norm training behavior)
       model.train()
       total_loss : float = 0
       correct : int = 0
       total : int = 0

       # Iterate through training batches
       for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}") :
           # Move batch data to computing device
           batch = {k : v.to(Config.DEVICE) for k, v in batch.items()}
           
           # Reset gradients from previous iteration
           optimizer.zero_grad()
           
           # Forward pass : compute loss and predictions
           loss, logits = model(**batch)
           
           # Backward pass : compute gradients
           loss.backward()
           
           # Clip gradients to prevent exploding gradients problem
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           
           # Update model parameters
           optimizer.step()

           # Accumulate training statistics
           total_loss += loss.item()
           preds = torch.argmax(logits, dim=1)
           correct += (preds == batch["labels"]).sum().item()
           total += batch["labels"].size(0)

       # Calculate training accuracy for current epoch
       train_acc : float = correct / total

       # Evaluate model on validation set
       val_acc : float = evaluate(model, val_loader)
       print(f"Epoch {epoch+1} : Train Acc = {train_acc :.4f} | Val Acc = {val_acc :.4f}")

       # Early stopping logic : save best model and track patience
       if val_acc > best_val_acc :
           best_val_acc = val_acc
           patience = 0
           # Create directory if it doesn't exist
           os.makedirs(os.path.dirname(Config.SAVE_PATH), exist_ok=True)
           torch.save(model.state_dict(), Config.SAVE_PATH)
       else :
           patience += 1
           # Stop training if validation performance doesn't improve
           if patience >= max_patience :
               print("Early stopping triggered")
               break

   # Final evaluation : load best model and test on unseen data
   model.load_state_dict(torch.load(Config.SAVE_PATH))
   test_acc : float = evaluate(model, test_loader, detailed=True)
   print(f"Final Test Accuracy : {test_acc :.4f}")

def evaluate(model : Module, dataloader : DataLoader, detailed : bool = False) -> float :
   """
   Evaluate model performance on given dataset.
   
   Inputs :
   ----------
   model : The trained neural network model
   dataloader : DataLoader containing evaluation data
   detailed : If True, prints detailed classification report
   
   Returns :
   ----------
   Overall accuracy score as float
   """
   # Set model to evaluation mode (disables dropout, uses running batch norm stats)
   model.eval()
   preds : List[int] = []
   labels : List[int] = []

   # Disable gradient computation for memory efficiency during evaluation
   with torch.no_grad() :
       for batch in dataloader :
           # Move batch to computing device
           batch = {k : v.to(Config.DEVICE) for k, v in batch.items()}
           
           # Forward pass to get predictions
           output = model(**batch)
           
           # Handle different output formats (with or without loss)
           logits = output[1] if isinstance(output, tuple) else output
           
           # Get predicted class (argmax over logits)
           pred = torch.argmax(logits, dim=1)
           
           # Store predictions and true labels for metric computation
           preds.extend(pred.cpu().numpy())
           labels.extend(batch["labels"].cpu().numpy())

   # Calculate overall accuracy
   acc : float = accuracy_score(labels, preds)

   # Print detailed metrics if requested
   if detailed :
       print("\nDetailed Classification Report :")
       print(classification_report(labels, preds, target_names=["Human", "AI Generated"]))

   return acc

if __name__ == "__main__" :
   """
   Main execution script for DistilBERT training and evaluation.
   
   This script trains a regularized DistilBERT model on economic text classification
   with early stopping and comprehensive evaluation metrics.
   """
   train_and_evaluate()