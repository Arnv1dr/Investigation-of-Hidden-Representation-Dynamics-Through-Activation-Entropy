# Global experiment configuration

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# Training seed controls model initialization and dataloader shuffling.
SEED = 42

# Split seed controls the fixed stratified train/validation split.
SPLIT_SEED = 42

VAL_PER_CLASS = 1000
NUM_WORKERS = 0

DATA_DIR = "./data"
SPLIT_DIR = "./splits"
RESULTS_DIR = "./results/logs"
TABLES_DIR = "./results/tables"

# Hidden post-activation layers monitored for activation entropy.
MLP_MONITORED_LAYERS = ["relu1", "relu2"]
CNN_MONITORED_LAYERS = ["relu1", "relu2", "relu3"]
