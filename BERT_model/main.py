import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import Module
from model import TransformerClassifier
from config import Config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
from typing import Tuple, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import BERTDataset, BERTData
from tqdm import tqdm

def train(model : Module, 
         train_loader : DataLoader, 
         val_loader : DataLoader, 
         epochs : int = Config.EPOCHS, 
         lr : float = Config.LEARNING_RATE, 
         device : str = Config.DEVICE) -> None :
   """
   Train the BERT model for classification.
   
   Inputs :
   ----------
   model : The neural network model to train
   train_loader : DataLoader containing training data
   val_loader : DataLoader containing validation data
   epochs : Number of training epochs (default from Config)
   lr : Learning rate for optimization (default from Config)
   device : Computing device ('cpu' or 'cuda', default from Config)
   """
   # Initialization of AdamW optimizer with specified learning rate
   optimizer = AdamW(model.parameters(), lr=lr)
   model.to(device)

   for epoch in range(epochs) :
       # Set model to training mode (enables dropout, batch norm training behavior)
       model.train()
       total_loss : float = 0
       correct : int = 0
       total : int = 0

       print(f"\nEpoch {epoch + 1}/{epochs}")

       # Training loop over batches
       for batch in tqdm(train_loader, desc="Training") :
           batch = {k : v.to(device) for k, v in batch.items()}
           # Reset gradients from previous iteration
           optimizer.zero_grad()
           # Forward pass : compute loss and logits
           loss, logits = model(**batch)
           # Backward pass : compute gradients
           loss.backward()
           # Update model parameters
           optimizer.step()
           # Loss and accuracy statistics
           total_loss += loss.item()
           predictions = torch.argmax(logits, dim=1)
           correct += (predictions == batch["labels"]).sum().item()
           total += batch["labels"].size(0)

       # Training accuracy and evaluation on validation set
       train_acc : float = correct / total
       val_acc : float = evaluate(model, val_loader, device)
       
       print(f"Epoch {epoch+1}/{epochs} | Loss : {total_loss :.4f} | Train Acc : {train_acc :.4f} | Val Acc : {val_acc :.4f}")
       
       # Save model state after each epoch
       torch.save(model.state_dict(), Config.SAVE_PATH)
       print(f"Model saved to {Config.SAVE_PATH}")

def evaluate(model : Module, 
            dataloader : DataLoader, 
            device : str = Config.DEVICE, 
            detailed : bool = False) -> float :
   """
   Evaluate model performance on a given dataset.
   
   Inputs :
   ----------
   model : The trained neural network model
   dataloader : DataLoader containing evaluation data
   device : Computing device ('cpu' or 'cuda', default from Config)
   detailed : If True, prints detailed classification metrics
   
   Returns :
   ----------
   Overall accuracy score
   """
   # Set model to evaluation mode (disables dropout, batch norm uses running stats)
   model.eval()
   all_preds : list = []
   all_labels : list = []

   # Disable gradient computation for efficiency during evaluation
   with torch.no_grad() :
       for batch in tqdm(dataloader, desc="Evaluating") :
           # Move batch to device
           batch = {k : v.to(device) for k, v in batch.items()}
           
           # Forward pass
           output = model(**batch)
           
           # Handle different model output formats (with or without loss)
           logits = output[1] if isinstance(output, tuple) else output

           # Get predicted class (highest probability)
           predictions = torch.argmax(logits, dim=1)
           
           # Store predictions and true labels for metric calculation
           all_preds.extend(predictions.cpu().numpy())
           all_labels.extend(batch["labels"].cpu().numpy())

   # Calculate overall accuracy
   accuracy : float = accuracy_score(all_labels, all_preds)
   
   # Print detailed metrics
   if detailed :
       print(f"\nDetailed Evaluation Results :")
       print(f"Accuracy : {accuracy :.4f}")
       print("\nClassification Report :")
       print(classification_report(all_labels, all_preds, 
                                 target_names=['Human', 'AI Generated']))
       print("\nConfusion Matrix :")
       print(confusion_matrix(all_labels, all_preds))
   
   return accuracy

if __name__ == "__main__" :
   """
   Main execution script for training and evaluating the BERT classifier.
   
   This script :
   1. Loads the preprocessed dataset
   2. Initializes the BERT-based transformer model
   3. Trains the model on training data with validation monitoring
   4. Evaluates final performance on the test set
   """
   # Load preprocessed dataset with train/validation/test splits
   data : BERTData = BERTData()

   # Initialize BERT-based classification model
   model : TransformerClassifier = TransformerClassifier()

   # Train the model with validation monitoring
   train(model, data.train_data, data.val_data)

   # Final evaluation on test set with detailed metrics
   print("\nFinal Test Evaluation :")
   test_acc : float = evaluate(model, data.test_data, detailed=True)
   print(f"Test Accuracy : {test_acc :.4f}")