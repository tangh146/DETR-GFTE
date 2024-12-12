import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from detr_data import create_dataloaders

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import state_dict_cleaner

class Detr(pl.LightningModule):

    def __init__(
            self,
            train_mode,
            pretrained, checkpoint,
            config,
            lr = 1e-4,
            lr_backbone = 1e-5,
            weight_decay = 1e-4):
        
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=pretrained, 
            num_labels=config['training']['num_detr_classes'],
            ignore_mismatched_sizes=True
        )
        if checkpoint:
            state_dict = torch.load(checkpoint)['state_dict']
            state_dict = state_dict_cleaner(state_dict)
            self.model.load_state_dict(state_dict)
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.image_processor = DetrImageProcessor.from_pretrained(pretrained)

        if train_mode:
            self.dataloaders = create_dataloaders(
                config['training']['train_data_path'],
                config['training']['train_labels_path'],
                config['training']['val_data_path'],
                config['training']['val_labels_path'],
                batch_size=config['training']['batch_size'],
                image_processor=self.image_processor)

        if self.train:
            self.model.train()
        else:
            self.model.eval()

    # expecting a cv2.imread numpy array
    def forward(self, image):

        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)

        inputs = self.image_processor(images=image, return_tensors='pt').to(self.device)

        detr_outputs = self.model(**inputs)

        return detr_outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        detr_outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        detr_loss = detr_outputs.loss
        loss_dict = detr_outputs.loss_dict
        
        return detr_loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx) 
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]