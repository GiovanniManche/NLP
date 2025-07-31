from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import pandas as pd, numpy as np
import os
import torch

class BERTDataset(Dataset):
    """
    Class that transforms raw data into something understandanble by the BERT model.
    Inputs : 
    ------------
        - dataframe : a dataframe that contains the text and the label (0 if not AI generated, 1 otherwise)
        - tokenizer : pretrained component from the transformers library that helps transforming raw 
                      text data into tokens, objects readable by the BERT model. A token is a piece of text 
                      that the model can understand and process. 
        - max_length : integer that set the maximal number of tokens for each sequence.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer: BertTokenizer, 
                 max_length: int=Config.MAX_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.max_length = max_length

    def __len__(self):
        """
        Compute the number of texts / labels in the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        """
        Retrieves the text at the idx place in the list of text.
        Then tokenizes the text and creates an attention mask (to differenciate between real tokens and padding tokens).
        Padding is used for the inputs to have the same token length (which is required in neural networks that process data in batches, like the BERT model) 
        if some are shorter than ohter. A padding token is ignored by the model. 
        """
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTData:
    """
    Handles the preparation of the dataset to train, validate and evaluate the model. 
    Inputs :
    ------------
        - csv_path : path of the CSV that contains the texts and the labels
        - tokenizer_name : specifies the pretrained tokenizer to use from the HuggingFace transformers library.
                           bert-base-uncased provides a good balance between performance and computation and the 
                           tokenizer doesn't differenciate between upper and lower cases.
        - batch_size : defines how many samples the model processes at once (batches) for training
        - eval_batch_size : same, but for evaluation. 
    """
    def __init__(self, csv_path: str = Config.CSV_PATH, 
                 tokenizer_name:str =Config.MODEL_NAME, 
                 batch_size:int = Config.BATCH_SIZE, 
                 eval_batch_size:int = Config.EVAL_BATCH_SIZE):
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found : {csv_path}")
        df = pd.read_csv(csv_path, sep = ";", encoding="ISO-8859-1")
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The CSV file must have the following columns : {required_columns}")
        df_train, df_val, df_test = self._split(df)

        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        self.train_data = DataLoader(BERTDataset(df_train, tokenizer), batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(BERTDataset(df_val, tokenizer), batch_size=eval_batch_size)
        self.test_data = DataLoader(BERTDataset(df_test, tokenizer), batch_size=eval_batch_size)



    def _split(self, df) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """
        Split the data into non-overlapping training, validation, and test sets.
        The splits are stratified based on the label column to preserve class proportions.
    
        Returns
        -------
        train : pd.DataFrame
            The training set (80% of total data)
        val : pd.DataFrame
            The validation set (10% of total data)
        test : pd.DataFrame
            The test set (10% of total data)
        """
        # First split : 90% train_val and 10% test
        train_val, test = train_test_split(
            df, test_size=0.1, stratify=df["label"], random_state=Config.RANDOM_STATE
        )

        # Second split : from train_val, we extract 1/9 for validation to get 80/10/10 overall
        train, val = train_test_split(
            train_val, test_size=1/9, stratify=train_val["label"], random_state=Config.RANDOM_STATE
        )

        return train, val, test

