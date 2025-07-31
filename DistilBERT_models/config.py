class Config :
   """
   Class to configure model hyperparameters,
   and path to load the database and export the model.
   Should be used for distilBERT.
   """
   # Model parameters
   # Model name (distilbert-base uncased for the DistilBERT model that doesn't differentiate
   # upper cases and lower cases)
   MODEL_NAME : str = "distilbert-base-uncased"
   # Max number of tokens for each text (reduced from BERT's 512 for efficiency)
   MAX_LENGTH : int = 256
   # First dropout rate applied after intermediate layer (high regularization)
   DROPOUT1 : float = 0.6
   # Second dropout rate applied before final classification layer
   DROPOUT2 : float = 0.4
   # Hidden dimension of intermediate classification layer
   INTERMEDIATE_DIM : int = 128
   
   # Training parameters
   # Number of samples processed together in each training step
   BATCH_SIZE : int = 16
   # Number of samples processed together during evaluation (can be larger)
   EVAL_BATCH_SIZE : int = 32
   # Step size for gradient descent optimization (lower than BERT for DistilBERT)
   LEARNING_RATE : float = 1e-5
   # Number of complete passes through the entire training dataset
   EPOCHS : int = 5
   # L2 regularization strength to prevent overfitting
   WEIGHT_DECAY : float = 0.01
   
   # Data parameters
   CSV_PATH = "economic_articles/clean_economic_dataset.csv"
   
   # Other parameters
   # Computing device ("cpu" or "cuda")
   DEVICE = "cpu"
   SAVE_PATH = "DistilBERT_models/saved_models/regularized_model.pth"