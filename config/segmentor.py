import pytorch_lightning as pl
import torch 
from loss.proposed_loss import Guide_Fusion_Loss
from metric.metrics import dice_score, iou_score
class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.automatic_optimization = True  # Enable PyTorch Lightning's automatic optimization
        
        # Print GPU usage information
        print(f"\n📊 Segmentor initialized. CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 Current CUDA device: {torch.cuda.current_device()}")
            print(f"📊 Device name: {torch.cuda.get_device_name(0)}\n")

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        
        # Ensure inputs are on the correct device
        device = self.device
        image = image.to(device)
        y_true = y_true.to(device)
        
        y_pred = self.model(image)
        
        # Ensure shapes match for loss calculation
        # If y_true has shape [batch_size, height, width], add channel dimension to make it [batch_size, 1, height, width]
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)
            
        loss = Guide_Fusion_Loss(y_pred, y_true)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou

    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice, "train_iou": iou}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"val_loss":loss, "val_dice": dice, "val_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss":loss, "test_dice": dice, "test_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=5)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers