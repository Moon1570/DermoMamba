# Training config
from config.segmentor import Segmentor
from config.data_config import train_dataset, test_dataset
import pytorch_lightning as pl
import os
from module.model.proposed_net import DermoMamba
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set random seed for reproducibility
pl.seed_everything(42)

# Initialize model
model = DermoMamba()
segmentor = Segmentor(model=model)

# Create output directory
os.makedirs('./weight/ISIC2018/', exist_ok=True)

# Callbacks
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(
    './weight/ISIC2018/', 
    filename="dermomamba-{epoch:02d}-{val_dice:.4f}",
    monitor="val_dice", 
    mode="max", 
    save_top_k=3,
    verbose=True, 
    save_weights_only=False,
    auto_insert_metric_name=False
)

early_stopping = EarlyStopping(
    monitor='val_dice',
    patience=15,
    mode='max',
    verbose=True
)

progress_bar = pl.callbacks.TQDMProgressBar()

# Logger
logger = TensorBoardLogger("tb_logs", name="dermomamba")

# Training parameters
PARAMS = {
    "benchmark": True, 
    "enable_progress_bar": True,
    "logger": logger,
    "callbacks": [check_point, early_stopping, progress_bar],
    "log_every_n_steps": 10, 
    "num_sanity_val_steps": 2, 
    "max_epochs": 200,
    "precision": 16,
    "accelerator": 'gpu' if pl.utilities.cuda.available() else 'cpu',
    "devices": 1
}

trainer = pl.Trainer(**PARAMS)

# CHECKPOINT_PATH = ""
# segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)

# Training
trainer.fit(segmentor, train_dataset, test_dataset)