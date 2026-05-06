import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import wandb
from tqdm import tqdm
from model import ConsensusModel_SingleTask  
from config import ModelConfig, TrainingConfig, DataConfig
from data_loader import create_data_loaders
from loss import NaNMSELoss
from trainer_stage2 import Stage2Trainer

class SingleTaskTrainer(Stage2Trainer):
    def __init__(self, model, config, product_name):

        self.product_name = product_name

        super().__init__(model, config, [product_name])

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'SingleTask {self.product_name} Epoch {epoch+1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            dynamic_input = batch['dynamic_features'].to(self.device)
            static_input = batch['static_features'].to(self.device)


            target_list = batch['product_targets'][self.product_name]
            target_tensor = torch.stack(target_list, dim=0).to(self.device)  # [B, 7]


            dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1,2)) | torch.isinf(dynamic_input).any(dim=(1,2)))
            static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
            target_valid = ~(torch.isnan(target_tensor).any(dim=1) | torch.isinf(target_tensor).any(dim=1))
            valid_mask = dynamic_valid & static_valid & target_valid
            if valid_mask.sum() == 0:
                continue

            dynamic = dynamic_input[valid_mask]
            static = static_input[valid_mask]
            target = target_tensor[valid_mask]

            pred, _ = self.model(dynamic, static)  

            loss = self.criterion(pred.float(), target.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            del dynamic_input, static_input, target_tensor
            torch.cuda.empty_cache()

        avg_loss = total_loss / min(len(train_loader), self.config.max_batches_per_epoch or len(train_loader))
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                dynamic_input = batch['dynamic_features'].to(self.device)
                static_input = batch['static_features'].to(self.device)

                target_list = batch['product_targets'][self.product_name]
                target_tensor = torch.stack(target_list, dim=0).to(self.device)

                dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1,2)) | torch.isinf(dynamic_input).any(dim=(1,2)))
                static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
                target_valid = ~(torch.isnan(target_tensor).any(dim=1) | torch.isinf(target_tensor).any(dim=1))
                valid_mask = dynamic_valid & static_valid & target_valid
                if valid_mask.sum() == 0:
                    continue

                dynamic = dynamic_input[valid_mask]
                static = static_input[valid_mask]
                target = target_tensor[valid_mask]

                pred, _ = self.model(dynamic, static)
                loss = self.criterion(pred.float(), target.float())
                total_loss += loss.item()

        avg_loss = total_loss / min(len(val_loader), self.config.max_batches_per_epoch or len(val_loader))
        return avg_loss


def main():
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    training_config.device = device


    train_loader, val_loader, test_loader, norm_params = create_data_loaders(
        data_config,
        training_config=training_config,
        grid_mask_path='./dataset/mask_Northeast_China.npy',
        normalize=True
    )


    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }

    for name, path in pretrained_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"model  {name} not found: {path}")


    products = ['era5', 'colm', 'smci']
    for prod in products:


        training_config.save_dir = f'./checkpoints/stage2_ablation_C/{prod}'


        model = ConsensusModel_SingleTask(
            config=training_config,
            pretrained_paths=pretrained_paths,
            target_name=prod,
            feature_dim=128,
            proj_dim=64,
            num_heads=4
        )


        trainer = SingleTaskTrainer(
            model=model,
            config=training_config,
            product_name=prod
        )

        best_val_loss = trainer.train(train_loader, val_loader)




if __name__ == "__main__":
    torch.manual_seed(7998)
    np.random.seed(7998)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        main()
    except KeyboardInterrupt:
        print("\n train break")
    except Exception as e:
        print(f"\n error: {e}")
        import traceback
        traceback.print_exc()
