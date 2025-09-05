"""
Data configuration for DermoMamba training
"""
from datasets.isic_dataset import create_data_loaders

# Data paths
DATA_ROOT = "D:/Research/DermoMamba/data/ISIC2018_proc"
TRAIN_SPLIT = "splits/isic2018_train.txt"
VAL_SPLIT = "splits/isic2018_val.txt"

# Training parameters
BATCH_SIZE = 4
NUM_WORKERS = 0  # Set to 0 for Windows to avoid multiprocessing issues
INPUT_SIZE = (192, 256)

def get_data_loaders():
    """Create and return train and validation data loaders"""
    train_loader, val_loader = create_data_loaders(
        data_root=DATA_ROOT,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    return train_loader, val_loader

# Export the data loaders
train_dataset, test_dataset = get_data_loaders()

print(f"Training samples: {len(train_dataset.dataset)}")
print(f"Validation samples: {len(test_dataset.dataset)}")
