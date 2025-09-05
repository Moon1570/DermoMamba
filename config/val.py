import pytorch_lightning as pl
from loss.proposed_loss import Guide_Fusion_Loss
from metric.metrics import dice_score, iou_score
from module.model.proposed_net import DermoMamba
from config.data_config import test_dataset
import torch

class ValidationSegmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = Guide_Fusion_Loss(y_pred, y_true)
        print(f"Loss: {loss.cpu().numpy():.4f}", end=' ')
        
        # Calculate Dice and IoU scores
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        
        # Compute TP, FP, FN, TN for Precision, Recall, and F-Score
        y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()  # Apply sigmoid and threshold
        TP = (y_pred_bin * y_true).sum(dim=(1, 2, 3))
        FP = ((y_pred_bin == 1) & (y_true == 0)).sum(dim=(1, 2, 3))
        FN = ((y_pred_bin == 0) & (y_true == 1)).sum(dim=(1, 2, 3))
        
        # Precision, Recall, and F-Score
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Average metrics over the batch
        precision_mean = precision.mean().item()
        recall_mean = recall.mean().item()
        f_score_mean = f_score.mean().item()
        
        # Log all metrics
        metrics = {
            "test_loss": loss,
            "test_dice": dice,
            "test_iou": iou,
            "test_precision": precision_mean,
            "test_recall": recall_mean,
            "test_f_score": f_score_mean,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

def run_validation(checkpoint_path):
    """Run validation on the test dataset"""
    model = DermoMamba()
    
    # Load model from checkpoint
    segmentor = ValidationSegmentor.load_from_checkpoint(checkpoint_path, model=model)
    
    # Set up trainer
    trainer = pl.Trainer(
        accelerator='gpu' if pl.utilities.cuda.available() else 'cpu',
        devices=1,
        logger=False
    )
    
    # Run validation
    print(f"Running validation with {len(test_dataset.dataset)} samples...")
    results = trainer.test(segmentor, test_dataset)
    return results

if __name__ == "__main__":
    # Example usage
    CHECKPOINT_PATH = "./weight/ISIC2018/dermomamba-best.ckpt"
    
    try:
        results = run_validation(CHECKPOINT_PATH)
        print("Validation completed successfully!")
        print("Results:", results)
    except FileNotFoundError:
        print(f"Checkpoint file not found: {CHECKPOINT_PATH}")
        print("Please train the model first or provide the correct checkpoint path.")