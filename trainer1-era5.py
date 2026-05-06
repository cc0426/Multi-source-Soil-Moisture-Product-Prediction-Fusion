# train_stage1.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from model import ProductModel
from config import ModelConfig, TrainingConfig, DataConfig
from data_loader import create_data_loaders
from loss import NaNMSELoss


class Stage1Trainer:
    def __init__(self, model, product_name, config):

        self.model = model
        self.product_name = product_name
        self.config = config
        self.device = config.device
        self.model.to(self.device)


        self.optimizer = self._create_optimizer()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=20
        )
        
        self.criterion = torch.nn.MSELoss()

        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }


        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.config.resume_checkpoint = "/home/zhangcheng/Soil_Moisture/CML_FD/checkpoints/stage1/colm/stage1_best_model.pth"  

        self.save_dir = Path(config.save_dir) / product_name
        self.save_dir.mkdir(parents=True, exist_ok=True)



    def _create_optimizer(self):
        if self.config.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(),
                              lr=self.config.learning_rate)
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(),
                               lr=self.config.learning_rate)
        else:
            return optim.SGD(self.model.parameters(),
                             lr=self.config.learning_rate)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'{self.product_name} Epoch {epoch+1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):

            dynamic_input = batch['dynamic_features'].to(self.device)
            static_input = batch['static_features'].to(self.device)

            target = torch.stack(batch['product_targets'][self.product_name], dim=0).to(self.device)  # [B, 7]
            dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1, 2)) | torch.isinf(dynamic_input).any(dim=(1, 2)))
            static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
            target_valid = ~(torch.isnan(target).any(dim=1) | torch.isinf(target).any(dim=1))
            valid_mask = dynamic_valid & static_valid & target_valid

            valid_count = valid_mask.sum().item()
            if valid_count == 0:

                continue


            dynamic = dynamic_input[valid_mask]
            static = static_input[valid_mask]
            target = target[valid_mask]

            pred, _ = self.model(dynamic, static)
            mask = ~torch.isnan(target).any(dim=1)
            if mask.sum() == 0:

                pbar.set_postfix({'loss': 'NaN', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
                continue

            pred_valid = pred[mask]
            target_valid = target[mask]

            # loss = NaNMSELoss.fit(None,pred_valid.float(), target_valid.float(),torch.nn.MSELoss())
            loss = NaNMSELoss.fit(None,pred_valid.float(), target_valid.float(),torch.nn.MSELoss())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()


            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            del dynamic_input, static_input, target, pred
            torch.cuda.empty_cache()

            if self.config.max_batches_per_epoch and batch_idx >= self.config.max_batches_per_epoch:
                break

        num_batches = min(len(train_loader), self.config.max_batches_per_epoch or len(train_loader))
        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'{self.product_name} Validation')
            for batch_idx, batch in enumerate(pbar):
                dynamic_input = batch['dynamic_features'].to(self.device)
                static_input = batch['static_features'].to(self.device)
                target = torch.stack(batch['product_targets'][self.product_name], dim=0).to(self.device)
                dynamic_valid = ~(
                            torch.isnan(dynamic_input).any(dim=(1, 2)) | torch.isinf(dynamic_input).any(dim=(1, 2)))
                static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
                target_valid = ~(torch.isnan(target).any(dim=1) | torch.isinf(target).any(dim=1))
                valid_mask = dynamic_valid & static_valid & target_valid

                valid_count = valid_mask.sum().item()
                if valid_count == 0:

                    continue


                dynamic = dynamic_input[valid_mask]
                static = static_input[valid_mask]
                target = target[valid_mask]
                pred, _ = self.model(dynamic, static)
                mask = ~torch.isnan(target).any(dim=1)
                if mask.sum() == 0:

                    pbar.set_postfix({'loss': 'NaN', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
                    continue

                pred_valid = pred[mask]
                target_valid = target[mask]

                loss = NaNMSELoss.fit(None,pred_valid.float(), target_valid.float(),torch.nn.MSELoss())
                #self.criterion(pred_valid, target_valid)
                #loss = self.criterion(pred_valid, target_valid)
                # loss = self.criterion(pred, target)

                total_loss += loss.item()

                del dynamic_input, static_input, target, pred
                torch.cuda.empty_cache()

                if self.config.max_batches_per_epoch and batch_idx >= self.config.max_batches_per_epoch:
                    break

        num_batches = min(len(val_loader), self.config.max_batches_per_epoch or len(val_loader))
        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'product_name': self.product_name,
            'best_val_loss': self.best_val_loss,       
            'patience_counter': self.patience_counter  
        }

        checkpoint_path = self.save_dir / f'stage1_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.save_dir / 'stage1_best_model.pth'
            torch.save(checkpoint, best_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)

        return checkpoint['epoch']

    def train(self, train_loader, val_loader):


        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.product_name}_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(self.config)
            )


        best_val_loss = self.best_val_loss
        patience_counter = self.patience_counter
        start_epoch = 0


        if hasattr(self.config, 'resume_checkpoint') and os.path.exists(self.config.resume_checkpoint):

            start_epoch = self.load_checkpoint(self.config.resume_checkpoint) + 1
            best_val_loss = self.best_val_loss
            patience_counter = self.patience_counter



        early_stop_patience = getattr(self.config, 'early_stop_patience', 200)
        early_stop_min_delta = getattr(self.config, 'early_stop_min_delta', 1e-6)
        early_stopping_threshold = getattr(self.config, 'early_stopping_threshold', None)

        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            train_loss = self.train_epoch(train_loader, epoch)
            
            val_loss = self.validate(val_loader)

            
            #self.scheduler.step(val_loss)


            self.train_history['epoch'].append(epoch+1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)


            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                print(f"new best model")
            else:
                patience_counter += 1
                is_best = False
                print(f"early stop: {patience_counter}/{early_stop_patience}")


            self.best_val_loss = best_val_loss
            self.patience_counter = patience_counter

            print(f"  train loss: {train_loss:.6f}")
            print(f"  val loss: {val_loss:.6f}")
            print(f"  lr: {self.optimizer.param_groups[0]['lr']:.6f}")


            if (epoch+1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best)


            if patience_counter >= early_stop_patience:

                break


            if early_stopping_threshold is not None and val_loss < early_stopping_threshold:

                break

            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter
                })


        final_path = self.save_dir / 'stage1_final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'product_name': self.product_name,
            'best_val_loss': best_val_loss
        }, final_path)


        if self.config.use_wandb:
            wandb.finish()

        return best_val_loss


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

    product_names = ['era5']
    best_val_losses = {}

    for product in product_names:

        model = ProductModel(
            input_dim=model_config.input_dim,    
            shared_dim=model_config.shared_dim,   
            output_days=7
        )


        trainer = Stage1Trainer(
            model=model,
            product_name=product,
            config=training_config
        )


        best_val_loss = trainer.train(train_loader, val_loader)
        best_val_losses[product] = best_val_loss




    for p, loss in best_val_losses.items():
        print(f"  {p} best val loss: {loss:.6f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
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
