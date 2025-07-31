import pandas as pd
import numpy as np
from scipy import stats
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Any

# Import your existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import BERTDataset

def truncate_to_words(text : str, max_words : int = 300) -> str :
    """
    Truncate text to maximum number of words.
    
    Inputs :
    ----------
    text : Input text to truncate
    max_words : Maximum number of words to keep
    
    Returns :
    ----------
    Truncated text string
    """
    # Split text into individual words
    words = text.split()
    
    # Return original text if already within limit
    if len(words) <= max_words :
        return text
    
    # Take only the first max_words words to reduce length bias
    truncated_words = words[:max_words]
    return ' '.join(truncated_words)

def create_truncated_dataset(csv_path : str, max_words : int = 300, output_path : Optional[str] = None) -> Tuple[str, Dict[str, Any]] :
    """
    Create a dataset with all texts truncated to max_words.
    
    Inputs :
    ----------
    csv_path : Path to original CSV dataset
    max_words : Maximum number of words per text
    output_path : Path for output file (optional)
    
    Returns :
    ----------
    Tuple containing output path and statistics dictionary
    """
    
    print(f"TRUNCATING DATASET TO {max_words} WORDS")
    print("="*50)
    
    # Load original dataset with proper encoding
    df = pd.read_csv(csv_path, sep=";", encoding="ISO-8859-1")
    print(f"Original dataset : {len(df)} texts")
    
    # Calculate statistics before truncation for comparison
    df['original_word_count'] = df['text'].str.split().str.len()
    df['original_char_count'] = df['text'].str.len()
    
    print(f"\nBEFORE TRUNCATION :")
    # Analyze length distribution by class (human vs AI)
    for label in [0, 1] :
        subset = df[df['label'] == label]
        class_name = "Human" if label == 0 else "AI"
        print(f"   {class_name} :")
        print(f"     Average words : {subset['original_word_count'].mean() :.1f}")
        print(f"     Average chars : {subset['original_char_count'].mean() :.1f}")
        print(f"     Min-Max words : {subset['original_word_count'].min()}-{subset['original_word_count'].max()}")
    
    # Statistical test to check for significant length differences before truncation
    human_words_before = df[df['label'] == 0]['original_word_count']
    ai_words_before = df[df['label'] == 1]['original_word_count']
    t_stat_before, p_value_before = stats.ttest_ind(human_words_before, ai_words_before)
    
    print(f"\nSignificant difference BEFORE : p-value = {p_value_before :.6f}")
    if p_value_before < 0.05 :
        print("     WARNING : SIGNIFICANT DIFFERENCE")
    else :
        print("     OK : No significant difference")
    
    # Apply truncation to all texts to reduce length bias
    print(f"\nTRUNCATION IN PROGRESS...")
    df['truncated_text'] = df['text'].apply(lambda x : truncate_to_words(x, max_words))
    
    # Calculate new statistics after truncation
    df['new_word_count'] = df['truncated_text'].str.split().str.len()
    df['new_char_count'] = df['truncated_text'].str.len()
    
    print(f"\nAFTER TRUNCATION :")
    # Analyze length distribution after truncation
    for label in [0, 1] :
        subset = df[df['label'] == label]
        class_name = "Human" if label == 0 else "AI"
        print(f"   {class_name} :")
        print(f"     Average words : {subset['new_word_count'].mean() :.1f}")
        print(f"     Average chars : {subset['new_char_count'].mean() :.1f}")
        print(f"     Min-Max words : {subset['new_word_count'].min()}-{subset['new_word_count'].max()}")
    
    # Statistical test to check if length bias was successfully reduced
    human_words_after = df[df['label'] == 0]['new_word_count']
    ai_words_after = df[df['label'] == 1]['new_word_count']
    t_stat_after, p_value_after = stats.ttest_ind(human_words_after, ai_words_after)
    
    print(f"\nSignificant difference AFTER : p-value = {p_value_after :.6f}")
    if p_value_after < 0.05 :
        print("     WARNING : STILL SIGNIFICANT DIFFERENCE")
    else :
        print("     OK : No more significant difference!")
    
    # Analyze which texts were affected by truncation
    texts_truncated = df[df['original_word_count'] > max_words]
    print(f"\nTRUNCATION IMPACT :")
    print(f"   Texts truncated : {len(texts_truncated)}/{len(df)} ({len(texts_truncated)/len(df)*100 :.1f}%)")
    
    # Show distribution of truncated texts by class
    if len(texts_truncated) > 0 :
        print(f"   Distribution of truncated texts :")
        truncated_by_class = texts_truncated['label'].value_counts()
        for label, count in truncated_by_class.items() :
            class_name = "Human" if label == 0 else "AI"
            pct = count / len(texts_truncated) * 100
            print(f"     {class_name} : {count} texts ({pct :.1f}%)")
    
    # Create final clean dataset with only text and label columns
    final_df = df[['truncated_text', 'label']].copy()
    final_df.columns = ['text', 'label']
    
    # Save truncated dataset to file
    if output_path is None :
        output_path = f"truncated_{max_words}words_dataset.csv"
    
    final_df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
    print(f"\nTruncated dataset saved : {output_path}")
   
    # Summary statistics
    print(f"\nSUMMARY  :")
    print(f"   Original dataset : {len(df)} texts")
    print(f"   Length difference BEFORE : p={p_value_before :.6f}")
    print(f"   Length difference AFTER : p={p_value_after :.6f}")
    print(f"   Texts affected by truncation : {len(texts_truncated)} ({len(texts_truncated)/len(df)*100 :.1f}%)")
    print(f"   Bias reduction : {'YES' if p_value_after > 0.05 else 'NO'}")
    
    # Return results for programmatic use
    return output_path, {
        'original_size' : len(df),
        'p_value_before' : p_value_before,
        'p_value_after' : p_value_after,
        'texts_truncated' : len(texts_truncated),
        'truncation_rate' : len(texts_truncated)/len(df)*100
    }

def distilbert_test(truncated_csv_path : str) -> Optional[float] :
    """
    Test DistilBERT model on truncated dataset.
    
    Inputs :
    ----------
    truncated_csv_path : Path to the truncated CSV dataset
    
    Returns :
    ----------
    Test accuracy if successful, None if error
    """
    print(f"\nQUICK DISTILBERT TEST ON TRUNCATED DATASET")
    print("="*50)
    
    try :
        # Import model components (placed in try block to handle import errors)
        from model import RegularizedDistilBERT
        from config import Config
        from transformers import DistilBertTokenizer
        
        # Load truncated dataset with proper encoding
        df_truncated = pd.read_csv(truncated_csv_path, sep=";", encoding="utf-8")
        
        # Split dataset for quick evaluation (80% train, 20% test)
        train_df, test_df = train_test_split(
            df_truncated, test_size=0.2, 
            stratify=df_truncated['label'], 
            random_state=42
        )
        
        print(f"Train : {len(train_df)}, Test : {len(test_df)}")
        
        # Initialize tokenizer for DistilBERT
        tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # Create datasets using existing BERTDataset class
        train_dataset = BERTDataset(train_df, tokenizer, max_length=Config.MAX_LENGTH)
        test_dataset = BERTDataset(test_df, tokenizer, max_length=Config.MAX_LENGTH)
        
        # Create data loaders for batch processing
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch for speed
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
        
        # Initialize model and optimizer
        model = RegularizedDistilBERT().to(Config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        
        print(f"\nQUICK TRAINING (1 epoch) :")
        
        # Single epoch training for quick evaluation
        model.train()
        total_loss : float = 0
        correct : int = 0
        total : int = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc="Training") :
            # Move batch to computing device
            batch = {k : v.to(Config.DEVICE) for k, v in batch.items()}
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss, logits = model(**batch)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model parameters
            optimizer.step()
            
            # Track training statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        
        # Calculate training accuracy
        train_acc : float = correct / total
        
        # Evaluation function for test set
        def evaluate_quick(model, dataloader) :
            """Quick evaluation on test set"""
            model.eval()  # Set to evaluation mode
            preds, labels = [], []
            
            # Disable gradient computation for efficiency
            with torch.no_grad() :
                for batch in tqdm(dataloader, desc="Testing") :
                    # Move batch to device
                    batch = {k : v.to(Config.DEVICE) for k, v in batch.items()}
                    
                    # Forward pass
                    output = model(**batch)
                    logits = output[1] if isinstance(output, tuple) else output
                    
                    # Get predictions
                    pred = torch.argmax(logits, dim=1)
                    preds.extend(pred.cpu().numpy())
                    labels.extend(batch["labels"].cpu().numpy())
            
            return accuracy_score(labels, preds), preds, labels
        
        # Evaluate on test set
        test_acc, predictions, true_labels = evaluate_quick(model, test_loader)
        
        print(f"\nRESULTS ON TRUNCATED DATASET :")
        print(f"   Train Accuracy : {train_acc :.4f}")
        print(f"   Test Accuracy : {test_acc :.4f}")
        
        # Detailed classification metrics
        print(f"\nClassification Report :")
        print(classification_report(true_labels, predictions, target_names=["Human", "AI Generated"]))
        
        # Create and display confusion matrix
        print(f"\nCONFUSION MATRIX :")
        cm = confusion_matrix(true_labels, predictions)
        print(f"                 PREDICTED")
        print(f"              Human    AI")
        print(f"ACTUAL Human    {cm[0,0] :2d}    {cm[0,1] :2d}     ({sum(cm[0,:])} total)")
        print(f"         AI     {cm[1,0] :2d}    {cm[1,1] :2d}     ({sum(cm[1,:])} total)")
        
        # Save confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Human', 'AI'], 
                    yticklabels=['Human', 'AI'])
        plt.title('Confusion Matrix - Truncated Dataset')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Ensure output directory exists
        os.makedirs('DistilBERT_models/plots', exist_ok=True)
        plt.savefig('DistilBERT_models/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Confusion matrix saved as 'DistilBERT_models/plots/confusion_matrix.png'")
        
        # Interpret results for bias reduction effectiveness
        if test_acc < 0.90 :
            print("\nMore realistic performance! Truncation effective.")
        elif test_acc < 0.95 :
            print("\nStill high performance, but improved.")
        else :
            print("\nPerformance still too high, other biases present.")
        
        return test_acc
        
    except Exception as e :
        # Handle any errors during testing
        print(f"Error during test : {e}")
        print(f"Make sure you're running from the correct directory with access to :")
        print(f"   - model.py (RegularizedDistilBERT)")
        print(f"   - config.py")
        print(f"   - data.py (BERTDataset)")
        return None

def compare_models_truncated_vs_original() -> None :
    """
    Compare performance on original vs truncated dataset.
    """
    print(f"\nCOMPARING ORIGINAL VS TRUNCATED DATASET PERFORMANCE")
    print("="*60)
    
    # Test model performance on original dataset
    print("Testing on ORIGINAL dataset...")
    original_acc = distilbert_test("economic_articles/clean_economic_dataset.csv")
    
    # Test model performance on truncated dataset
    print("\nTesting on TRUNCATED dataset...")
    truncated_acc = distilbert_test("truncated_300words_dataset.csv")
    
    # Compare results if both tests successful
    if original_acc and truncated_acc :
        print(f"\nCOMPARISON RESULTS :")
        print(f"   Original dataset accuracy :  {original_acc :.4f}")
        print(f"   Truncated dataset accuracy : {truncated_acc :.4f}")
        print(f"   Performance drop : {original_acc - truncated_acc :.4f}")
        
        # Assess effectiveness of truncation for bias reduction
        if (original_acc - truncated_acc) > 0.05 :
            print("   Significant performance drop - bias reduction successful!")
        else :
            print("   Small performance change - other biases may remain.")

if __name__ == "__main__" :
    """
    Main execution script for dataset truncation and testing.
    
    This script :
    1. Truncates the original dataset to 300 words maximum
    2. Analyzes statistical differences before and after truncation
    3. Optionally tests DistilBERT performance on truncated data
    4. Compares original vs truncated dataset performance
    """
    print("DATASET TRUNCATION TOOL")
    print("="*60)
    
    # Verify that original dataset file exists
    original_path : str = "economic_articles/clean_economic_dataset.csv"
    if not os.path.exists(original_path) :
        print(f"Original dataset not found : {original_path}")
        print("Please make sure the file exists and run from the correct directory.")
        exit(1)
    
    # Step 1: Truncate the dataset to reduce length bias
    print("STEP 1 : Truncating dataset...")
    truncated_file, stats = create_truncated_dataset(
        original_path, 
        max_words=300,
        output_path="truncated_300words_dataset.csv"
    )
    
    # Optional: Quick test with DistilBERT
    print(f"\n" + "="*60)
    user_input = input("Do you want to test with DistilBERT now? (y/n) : ")
    if user_input.lower() == 'y' :
        bert_acc = distilbert_test(truncated_file)
    
    # Optional: Compare original vs truncated performance
    print(f"\n" + "="*60)
    user_input = input("Do you want to compare original vs truncated performance? (y/n) : ")
    if user_input.lower() == 'y' :
        compare_models_truncated_vs_original()
    
    # Final summary
    print(f"\nTRUNCATION COMPLETE!")
    print(f"Truncated dataset saved as : {truncated_file}")
    print(f"Use this file in your config.py : CSV_PATH = \"{truncated_file}\"")