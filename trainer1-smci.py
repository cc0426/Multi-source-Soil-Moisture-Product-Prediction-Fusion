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

# -------------------- 第一阶段训练器（单产品） --------------------
class Stage1Trainer:
    def __init__(self, model, product_name, config):
        """
        Args:
            model: ProductModel 实例
            product_name: 'era5', 'colm', 或 'smci'
            config: TrainingConfig 实例
        """
        self.model = model
        self.product_name = product_name
        self.config = config
        self.device = config.device
        self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=20
        )
        # 损失函数（简单 MSE）
        
        self.criterion = torch.nn.MSELoss()

        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }

        # 早停相关变量（初始化）
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.config.resume_checkpoint = "/home/zhangcheng/Soil_Moisture/CML_FD/checkpoints/stage1/smci/stage1_best_model.pth"  # 恢复训练的检查点路径

        # 保存目录（按产品分开）
        self.save_dir = Path(config.save_dir) / product_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"产品 {product_name} 的训练器初始化完成")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
            # 数据
            dynamic_input = batch['dynamic_features'].to(self.device)
            static_input = batch['static_features'].to(self.device)
            # 目标：只取当前产品的标签
            target = torch.stack(batch['product_targets'][self.product_name], dim=0).to(self.device)  # [B, 7]
            dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1, 2)) | torch.isinf(dynamic_input).any(dim=(1, 2)))
            static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
            target_valid = ~(torch.isnan(target).any(dim=1) | torch.isinf(target).any(dim=1))
            valid_mask = dynamic_valid & static_valid & target_valid

            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                print(f"Batch {batch_idx} 无有效样本，跳过")
                continue

            # 步骤2：根据掩码筛选样本
            dynamic = dynamic_input[valid_mask]
            static = static_input[valid_mask]
            target = target[valid_mask]
            # 前向
            pred, _ = self.model(dynamic, static)
            mask = ~torch.isnan(target).any(dim=1)
            if mask.sum() == 0:
                # 没有有效样本，跳过
                pbar.set_postfix({'loss': 'NaN', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
                continue

            pred_valid = pred[mask]
            target_valid = target[mask]
            # 损失
            # loss = NaNMSELoss.fit(None,pred_valid.float(), target_valid.float(),torch.nn.MSELoss())
            loss = NaNMSELoss.fit(None,pred_valid.float(), target_valid.float(),torch.nn.MSELoss())
            # 反向
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # 内存清理
            del dynamic_input, static_input, target, pred
            torch.cuda.empty_cache()

            # 限制 batch 数（可选）
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
                    print(f"Batch {batch_idx} 无有效样本，跳过")
                    continue

                # 步骤2：根据掩码筛选样本
                dynamic = dynamic_input[valid_mask]
                static = static_input[valid_mask]
                target = target[valid_mask]
                pred, _ = self.model(dynamic, static)
                mask = ~torch.isnan(target).any(dim=1)
                if mask.sum() == 0:
                    # 没有有效样本，跳过
                    pbar.set_postfix({'loss': 'NaN', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
                    continue

                pred_valid = pred[mask]
                target_valid = target[mask]
                # 损失
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
            'best_val_loss': self.best_val_loss,       # 保存当前最佳验证损失
            'patience_counter': self.patience_counter  # 保存当前早停计数器
        }
        # 最新检查点
        checkpoint_path = self.save_dir / f'stage1_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.save_dir / 'stage1_best_model.pth'
            torch.save(checkpoint, best_path)

        print(f"检查点已保存到: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        print(f"从 epoch {checkpoint['epoch']} 恢复训练（产品 {self.product_name}）")
        print(f"当前最佳验证损失: {self.best_val_loss:.6f}, 早停计数器: {self.patience_counter}")
        return checkpoint['epoch']

    def train(self, train_loader, val_loader):
        print("=" * 60)
        print(f"开始训练产品模型：{self.product_name}")
        print("=" * 60)

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.product_name}_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(self.config)
            )

        # 从实例变量初始化早停状态
        best_val_loss = self.best_val_loss
        patience_counter = self.patience_counter
        start_epoch = 0

        # 恢复检查点（如果有）
        if hasattr(self.config, 'resume_checkpoint') and os.path.exists(self.config.resume_checkpoint):
            # 注意：这里假设 resume_checkpoint 是完整的 .pth 文件路径
            # 如果需要针对不同产品自动拼接，可以在外部处理
            start_epoch = self.load_checkpoint(self.config.resume_checkpoint) + 1
            best_val_loss = self.best_val_loss
            patience_counter = self.patience_counter
            print(f"恢复训练，起始 epoch: {start_epoch+1}")

        # 获取早停相关参数（如果配置中没有则设置默认值）
        early_stop_patience = getattr(self.config, 'early_stop_patience', 200)
        early_stop_min_delta = getattr(self.config, 'early_stop_min_delta', 1e-6)
        early_stopping_threshold = getattr(self.config, 'early_stopping_threshold', None)

        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            # 验证
            val_loss = self.validate(val_loader)

            # 学习率调整
            #self.scheduler.step(val_loss)

            # 记录历史
            self.train_history['epoch'].append(epoch+1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)

            # 判断验证损失是否显著改善
            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                print(f"✨ 新的最佳模型！")
            else:
                patience_counter += 1
                is_best = False
                print(f"早停计数器: {patience_counter}/{early_stop_patience}")

            # 更新实例变量，以便保存检查点时能保存最新状态
            self.best_val_loss = best_val_loss
            self.patience_counter = patience_counter

            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存检查点
            if (epoch+1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best)

            # 早停条件1：基于耐心（patience）
            if patience_counter >= early_stop_patience:
                print(f"验证损失连续 {patience_counter} 个 epoch 未改善，提前停止训练。")
                break

            # 早停条件2：基于阈值（如果配置中设置了）
            if early_stopping_threshold is not None and val_loss < early_stopping_threshold:
                print(f"验证损失达到阈值 {early_stopping_threshold}，提前停止。")
                break

            # wandb 日志
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter
                })

        # 保存最终模型
        final_path = self.save_dir / 'stage1_final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'product_name': self.product_name,
            'best_val_loss': best_val_loss
        }, final_path)
        print(f"最终模型保存到: {final_path}")
        print(f"最佳验证损失: {best_val_loss:.6f}")

        if self.config.use_wandb:
            wandb.finish()

        return best_val_loss

# -------------------- 主函数 --------------------
def main():
    # 配置
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    training_config.device = device
    print(f"使用设备: {device}")

    # 数据加载器（一次性加载，后续分产品使用）
    print("加载数据集...")
    train_loader, val_loader, test_loader, norm_params = create_data_loaders(
        data_config,
        training_config=training_config,
        grid_mask_path='./dataset/mask_Northeast_China.npy',
        normalize=True
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    # 产品列表
    product_names = ['smci']
    best_val_losses = {}

    for product in product_names:
        print("\n" + "="*60)
        print(f"初始化产品模型: {product}")
        print("="*60)

        # 创建模型（ProductModel）
        model = ProductModel(
            input_dim=model_config.input_dim,      # 动态特征维度
            shared_dim=model_config.shared_dim,    # 时序特征提取器输出的维度
            output_days=7
        )

        # 训练器
        trainer = Stage1Trainer(
            model=model,
            product_name=product,
            config=training_config
        )

        # 训练
        best_val_loss = trainer.train(train_loader, val_loader)
        best_val_losses[product] = best_val_loss

        # 可选：在测试集上评估
        # test_loss = trainer.validate(test_loader)
        # print(f"产品 {product} 测试损失: {test_loss:.6f}")

    print("\n所有产品训练完成！")
    for p, loss in best_val_losses.items():
        print(f"  {p} 最佳验证损失: {loss:.6f}")

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
        print("\n训练中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()