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
from model import ConsensusModel_SingleTask   # 导入单任务模型
from config import ModelConfig, TrainingConfig, DataConfig
from data_loader import create_data_loaders
from loss import NaNMSELoss
from trainer_stage2 import Stage2Trainer
# 训练器类需要稍作修改，因为单任务模型返回的是 (pred, consensus_feat) 而不是字典
# 我们可以在循环中分别训练，每个目标使用自己的训练器实例
# 这里我们复用 Stage2Trainer，但需注意 forward 返回格式不同：单任务返回 pred 和 feat，不是字典
# 因此需要调整训练器的 train_epoch 和 validate 方法，或者为单任务单独写一个简单训练器。
# 为了简单，我们可以为单任务编写一个简单的训练循环，不依赖 Stage2Trainer。
# 但为了保持一致性，我们可以创建一个 SingleTaskTrainer 继承 Stage2Trainer 并覆盖相关方法。

class SingleTaskTrainer(Stage2Trainer):
    def __init__(self, model, config, product_name):
        # product_name 是当前训练的目标，如 'era5'
        self.product_name = product_name
        # 调用父类初始化，但 product_names 我们设为 [product_name]
        super().__init__(model, config, [product_name])

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'SingleTask {self.product_name} Epoch {epoch+1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            dynamic_input = batch['dynamic_features'].to(self.device)
            static_input = batch['static_features'].to(self.device)

            # 只取当前产品的标签
            target_list = batch['product_targets'][self.product_name]
            target_tensor = torch.stack(target_list, dim=0).to(self.device)  # [B, 7]

            # 有效样本筛选（与之前相同，但只检查当前产品）
            dynamic_valid = ~(torch.isnan(dynamic_input).any(dim=(1,2)) | torch.isinf(dynamic_input).any(dim=(1,2)))
            static_valid = ~(torch.isnan(static_input).any(dim=1) | torch.isinf(static_input).any(dim=1))
            target_valid = ~(torch.isnan(target_tensor).any(dim=1) | torch.isinf(target_tensor).any(dim=1))
            valid_mask = dynamic_valid & static_valid & target_valid
            if valid_mask.sum() == 0:
                continue

            dynamic = dynamic_input[valid_mask]
            static = static_input[valid_mask]
            target = target_tensor[valid_mask]

            # 前向
            pred, _ = self.model(dynamic, static)  # 单任务模型返回 (pred, consensus_feat)

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

    # 预训练模型路径
    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }

    for name, path in pretrained_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"预训练模型 {name} 未找到: {path}")

    # 对每个产品分别训练
    products = ['era5', 'colm', 'smci']
    for prod in products:
        print(f"\n{'='*60}")
        print(f"训练单任务模型: {prod}")
        print(f"{'='*60}")

        # 设置保存路径，每个产品一个子目录
        training_config.save_dir = f'./checkpoints/stage2_ablation_C/{prod}'

        # 创建单任务模型
        model = ConsensusModel_SingleTask(
            config=training_config,
            pretrained_paths=pretrained_paths,
            target_name=prod,
            feature_dim=128,
            proj_dim=64,
            num_heads=4
        )

        # 训练器
        trainer = SingleTaskTrainer(
            model=model,
            config=training_config,
            product_name=prod
        )

        # 开始训练
        best_val_loss = trainer.train(train_loader, val_loader)
        print(f"产品 {prod} 训练完成，最佳验证损失: {best_val_loss:.6f}")

    print("所有单任务模型训练完成。")


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