class Config:
    """
    Class to configure model hyperparameters, 
    and path to load the database and export the model. 
    Should be used for the standard BERT.
    """
    # Model parameters
    # Model name (bert-base uncased for the standard BERT model that doesn't differenciate
    # upper cases and lower cases)
    MODEL_NAME: str = "bert-base-uncased"
    # Max number of tokens for each text
    MAX_LENGTH: int = 512
    # % of neurons to drop at each training step (limits overfitting)
    DROPOUT: float = 0.1
    
    # Training parameters
    # Number of samples processed together in each training step
    BATCH_SIZE: int = 16
    # Number of samples processed together during evaluation
    EVAL_BATCH_SIZE: int = 32
    # Step size for gradient descent optimization (standard for BERT fine-tuning)
    LEARNING_RATE: float = 2e-5
    # Number of complete passes through the entire training dataset
    EPOCHS: int = 3
    
    # Data parameters
    CSV_PATH: str = "economic_articles/clean_economic_dataset.csv"
    
    # Other parameters
    # DEVICE : cpu or cuda
    DEVICE: str = "cpu"
    SAVE_PATH: str = "BERT_model/saved_models/bert_classifier.pth"
    # seed for reproductibility
    RANDOM_STATE: int = 42