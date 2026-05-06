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
from model import ConsensusModel
from config import ModelConfig, TrainingConfig, DataConfig
from data_loader import create_data_loaders
from loss import NaNMSELoss   # 假设已有此损失

class Stage2Trainer:
    def __init__(self, model, config, product_names):
        self.model = model
        self.config = config
        self.product_names = product_names
        self.device = config.device
        self.model.to(self.device)

        # 优化器（只训练融合部分和预测头，不训练冻结的特征提取器）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(trainable_params)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10
        )
        self.criterion = torch.nn.MSELoss()  # 基本 MSE，配合 NaNMSELoss 包装

        # 训练历史
        self.train_history = {'epoch': [], 'train_loss': [], 'val_loss': []}

        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 保存目录
        self.save_dir = Path(config.save_dir) / 'stage2'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"阶段二训练器初始化完成，可训练参数: {sum(p.numel() for p in trainable_params):,}")

    def _create_optimizer(self, params):
        if self.config.optimizer_type == 'adam':
            return optim.Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(params, lr=self.config.learning_rate)
        else:
            return optim.SGD(params, lr=self.config.learning_rate)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Stage2 Epoch {epoch+1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            dynamic_input = batch['dynamic_features'].to(self.device)
            static_input = batch['static_features'].to(self.device)

            # 收集三个产品的标签（每个是 [B, 7] 的列表，需要堆叠）
            targets = {}
            for name in self.product_names:
                # batch['product_targets'][name] 是包含7个标量的列表
                # 需要将其转换为 [B, 7] 的张量
                target_list = batch['product_targets'][name]
                target_tensor = torch.stack(target_list, dim=0).to(self.device)  # [B, 7]
                targets[name] = target_tensor

            # 有效样本筛选（任何产品标签或输入无效则跳过）
            dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1,2)) | torch.isinf(dynamic_input).any(dim=(1,2)))
            static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
            # 合并所有产品标签的有效性（至少有一个产品标签有效？我们要求所有产品都有效，因为是多任务）
            # 这里简单要求三个产品都有效，否则跳过整个样本
            target_valid = torch.ones_like(dynamic_valid, dtype=torch.bool)
            for name in self.product_names:
                target_valid &= ~(torch.isnan(targets[name]).any(dim=1) | torch.isinf(targets[name]).any(dim=1))

            valid_mask = dynamic_valid & static_valid & target_valid
            if valid_mask.sum() == 0:
                continue

            dynamic = dynamic_input[valid_mask]
            static = static_input[valid_mask]
            targets_valid = {name: targets[name][valid_mask] for name in self.product_names}

            # 前向
            preds, consensus_feat = self.model(dynamic, static)

            # 计算损失（三个产品的 MSE 之和）
            loss = 0.0
            for name in self.product_names:
                # 对每个产品，再次检查该样本的标签是否有效（但前面已整体过滤，可再细粒度处理）
                # 这里简单假设都有效
                loss += self.criterion(preds[name].float(), targets_valid[name].float())

            # 可选：多样性损失
            if getattr(self.config, 'use_diversity_loss', False) and hasattr(self.model, 'get_diversity_loss'):
                # 需要获取投影后的特征，这里简化，可以修改模型以返回中间特征
                # 这里跳过，或者模型内部实现
                pass

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 清理
            del dynamic_input, static_input, targets
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
                targets = {}
                for name in self.product_names:
                    target_list = batch['product_targets'][name]
                    target_tensor = torch.stack(target_list, dim=0).to(self.device)
                    targets[name] = target_tensor

                # 有效样本筛选（与训练相同）
                dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1,2)) | torch.isinf(dynamic_input).any(dim=(1,2)))
                static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
                target_valid = torch.ones_like(dynamic_valid, dtype=torch.bool)
                for name in self.product_names:
                    target_valid &= ~(torch.isnan(targets[name]).any(dim=1) | torch.isinf(targets[name]).any(dim=1))
                valid_mask = dynamic_valid & static_valid & target_valid
                if valid_mask.sum() == 0:
                    continue

                dynamic = dynamic_input[valid_mask]
                static = static_input[valid_mask]
                targets_valid = {name: targets[name][valid_mask] for name in self.product_names}

                preds, _ = self.model(dynamic, static)
                loss = 0.0
                for name in self.product_names:
                    loss += self.criterion(preds[name].float(), targets_valid[name].float())
                total_loss += loss.item()

        avg_loss = total_loss / min(len(val_loader), self.config.max_batches_per_epoch or len(val_loader))
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
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        checkpoint_path = self.save_dir / f'stage2_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.save_dir / 'stage2_best_model.pth'
            torch.save(checkpoint, best_path)
        print(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        print(f"从 epoch {checkpoint['epoch']} 恢复训练")
        return checkpoint['epoch']

    def train(self, train_loader, val_loader):
        print("="*60)
        print("开始训练阶段二共识模型")
        print("="*60)

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(self.config)
            )

        start_epoch = 0
        if hasattr(self.config, 'resume_checkpoint') and os.path.exists(self.config.resume_checkpoint):
            start_epoch = self.load_checkpoint(self.config.resume_checkpoint) + 1

        early_stop_patience = getattr(self.config, 'early_stop_patience', 50)  # 可以调小
        early_stop_min_delta = getattr(self.config, 'early_stop_min_delta', 1e-6)

        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.train_history['epoch'].append(epoch+1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)

            if val_loss < self.best_val_loss - early_stop_min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
                print("✨ 新的最佳模型！")
            else:
                self.patience_counter += 1
                is_best = False
                print(f"早停计数器: {self.patience_counter}/{early_stop_patience}")

            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            if (epoch+1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best)

            if self.patience_counter >= early_stop_patience:
                print(f"早停触发，结束训练")
                break

            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'best_val_loss': self.best_val_loss
                })

        final_path = self.save_dir / 'stage2_final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss
        }, final_path)
        print(f"最终模型保存到: {final_path}")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")

        if self.config.use_wandb:
            wandb.finish()

        return self.best_val_loss


def main():
    # 配置（与阶段一相同，但可以调整学习率等）
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    training_config.device = device
    print(f"使用设备: {device}")

    # 数据加载器
    print("加载数据集...")
    train_loader, val_loader, test_loader, norm_params = create_data_loaders(
        data_config,
        training_config=training_config,
        grid_mask_path='./dataset/mask_Northeast_China.npy',
        normalize=True
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    # 预训练模型的路径（请根据实际情况修改）
    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }

    # 检查路径是否存在
    for name, path in pretrained_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"预训练模型 {name} 未找到: {path}")

    # 创建共识模型
    # 注意：feature_dim 应与 ProductModel 的 shared_dim 一致（配置中为128）
    consensus_model = ConsensusModel(
        config=training_config,
        pretrained_paths=pretrained_paths,
        feature_dim=128,
        proj_dim=64,
        num_heads=4
    )

    # 训练器
    trainer = Stage2Trainer(
        model=consensus_model,
        config=training_config,
        product_names=['era5', 'colm', 'smci']
    )

    # 开始训练
    best_val_loss = trainer.train(train_loader, val_loader)
    print(f"阶段二训练完成，最佳验证损失: {best_val_loss:.6f}")


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
        print("\n训练中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()