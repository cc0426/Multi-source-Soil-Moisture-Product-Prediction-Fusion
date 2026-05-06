# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import DataConfig, ModelConfig, TrainingConfig
from data_loader import create_data_loaders, SoilMoistureDataset


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, model_type='transformer'):
        super().__init__()
        self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=128,
                num_layers=1,
                batch_first=True
            )
        # 时序特征压缩：将365天特征压缩为固定长度
        # self.dense = nn.Linear(12, 128)
        # self.temporal_pooling = LightweightTemporalPooling()

    def forward(self, x_time, x_static):
        batch_size, seq_len, _ = x_time.shape
        static_expanded = x_static.unsqueeze(1).expand(-1, seq_len, -1)  # [512, 365, 5]

        # 在特征维度上拼接
        x = torch.cat([x_time, static_expanded], dim=-1)  # [512, 365, 12]

        # 时序特征提取
        temporal_features,_ = self.encoder(x)  # [batch, 365, d_model]
        features = temporal_features[:,-1,:]

        return features

class ProductModel(nn.Module):
    def __init__(self, input_dim, shared_dim, output_days=7):
        super().__init__()
        self.temporal_extractor = TemporalFeatureExtractor(input_dim=input_dim)
        self.predictor = nn.Linear(shared_dim, output_days)  # 可换成更复杂的头

    def forward(self, dynamic_input, static_input):
        features = self.temporal_extractor(dynamic_input, static_input)  # [B, shared_dim]
        # if torch.isnan(features).any() or torch.isinf(features).any():
        #     print("NaN/Inf detected after temporal_extractor")
        pred = self.predictor(features)  # [B, 7]
        # if torch.isnan(pred).any() or torch.isinf(pred).any():
        #     print("NaN/Inf detected after predictor")
        return pred, features



class FrozenFeatureExtractor(nn.Module):
    """包装预训练的特征提取器，冻结参数，并确保输出为 [B, feat_dim]"""
    def __init__(self, product_model_path, device, feature_dim=128):
        super().__init__()
        # 加载完整 ProductModel
        checkpoint = torch.load(product_model_path, map_location=device, weights_only=False)
        # 注意：这里假设 checkpoint 中包含 'model_state_dict'
        # 需要根据你的实际保存格式调整
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # 直接是状态字典

        # 创建 ProductModel 实例并加载权重
        self.extractor = ProductModel(input_dim=12, shared_dim=feature_dim, output_days=7)
        self.extractor.load_state_dict(state_dict, strict=False)  # strict=False 忽略预测头不匹配
        # 冻结所有参数
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.extractor.eval()  # 设为评估模式

    def forward(self, dynamic_input, static_input):
        # ProductModel 的 forward 返回 (pred, features)
        # 我们只需要 features
        _, features = self.extractor(dynamic_input, static_input)
        return features


class ConsensusModel(nn.Module):
    """
    共识融合模型
    - 使用三个预训练的特征提取器（冻结）
    - 将三个特征投影到相同维度（如果特征维度不同）
    - 交叉注意力融合
    - 用共识特征分别预测三个产品的标签
    """
    def __init__(self, config, pretrained_paths, feature_dim=128, proj_dim=64, num_heads=4):
        """
        Args:
            config: 配置对象（包含 device 等）
            pretrained_paths: dict, 例如 {'era5': 'path/to/era5.pth', ...}
            feature_dim: 预训练特征提取器输出的维度（应与 ProductModel 的 shared_dim 一致）
            proj_dim: 投影后的统一维度
            num_heads: 交叉注意力头数
        """
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())

        # 加载三个冻结的特征提取器
        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        # 投影层（可选，如果特征维度已经一致且不需要降维，可以省略）
        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim)
            for name in self.product_names
        })

        # 交叉注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True   # 关键修改
        )

        # 融合后的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)   # 共识特征维度
        )

        # 三个产品的预测头（输入共识特征，输出7天）
        self.prediction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            ) for name in self.product_names
        })

        # 可选的多样性损失（在训练时计算）
        self.use_diversity_loss = getattr(config, 'use_diversity_loss', False)

    def forward(self, dynamic_input, static_input, return_attention=False):
        """
        前向传播
        Args:
            dynamic_input: [B, 365, n_forcing]
            static_input: [B, d_static]
            return_attention: 是否返回注意力权重
        Returns:
            preds: dict {name: [B, 7]}
            consensus_feat: [B, 128] 共识特征
            attn_weights: 可选，注意力权重
        """
        # 提取每个产品的特征
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)  # [B, feature_dim]
            feat = self.proj[name](feat)                               # [B, proj_dim]
            feats.append(feat)

        # 堆叠成 [B, 3, proj_dim]
        stacked = torch.stack(feats, dim=1)

        # 交叉注意力（自注意力，Q=K=V=stacked）
        attn_out, attn_weights = self.cross_attention(stacked, stacked, stacked,average_attn_weights=False)  # [B, 3, proj_dim]

        # 融合：对注意力输出取平均（或加权平均，这里简单平均）
        fused = attn_out.mean(dim=1)  # [B, proj_dim]

        # 共识特征
        consensus_feat = self.fusion_mlp(fused)  # [B, 128]

        # 预测
        preds = {}
        for name in self.product_names:
            preds[name] = self.prediction_heads[name](consensus_feat)  # [B, 7]

        if return_attention:
            return preds, consensus_feat, attn_weights
        return preds, consensus_feat

    def get_diversity_loss(self, product_features):
        """
        计算产品特征之间的多样性损失（可选）
        product_features: list of [B, proj_dim]
        """
        if len(product_features) < 2:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        n_pairs = 0
        for i in range(len(product_features)):
            for j in range(i+1, len(product_features)):
                sim = F.cosine_similarity(product_features[i], product_features[j], dim=-1).mean()
                loss += sim
                n_pairs += 1
        # 希望相似度小，所以返回相似度均值（作为损失的一部分，乘以一个权重）
        return loss / n_pairs

class ConsensusModel_MeanFusion(nn.Module):
    """
    消融实验 A：移除交叉注意力，直接平均三个投影后的特征。
    """
    def __init__(self, config, pretrained_paths, feature_dim=128, proj_dim=64):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())

        # 冻结的特征提取器
        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        # 投影层
        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim) for name in self.product_names
        })

        # 融合 MLP（输入为 proj_dim，因为平均后仍是 proj_dim）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # 三个产品的预测头
        self.prediction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            ) for name in self.product_names
        })

    def forward(self, dynamic_input, static_input):
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)  # [B, feature_dim]
            feat = self.proj[name](feat)                               # [B, proj_dim]
            feats.append(feat)

        # 直接平均（无注意力）
        stacked = torch.stack(feats, dim=1)   # [B, 3, proj_dim]
        fused = stacked.mean(dim=1)           # [B, proj_dim]

        consensus_feat = self.fusion_mlp(fused)  # [B, 128]

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat

class ConsensusModel_ConcatFusion(nn.Module):
    """
    消融实验 B：将三个投影后的特征拼接，再通过 MLP 融合。
    """
    def __init__(self, config, pretrained_paths, feature_dim=128, proj_dim=64):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())

        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim) for name in self.product_names
        })

        # 融合 MLP 输入维度为 proj_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.prediction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            ) for name in self.product_names
        })

    def forward(self, dynamic_input, static_input):
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)
            feat = self.proj[name](feat)
            feats.append(feat)

        # 拼接
        fused = torch.cat(feats, dim=-1)      # [B, proj_dim * 3]

        consensus_feat = self.fusion_mlp(fused)  # [B, 128]

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat

class ConsensusModel_SingleTask(nn.Module):
    """
    消融实验 C：单任务模型，仅预测一个目标产品。
    需为每个产品单独实例化并训练。
    """
    def __init__(self, config, pretrained_paths, target_name, feature_dim=128, proj_dim=64, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())
        self.target = target_name      # 目标产品名，如 'era5'

        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim) for name in self.product_names
        })

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # 仅一个预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, dynamic_input, static_input):
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)
            feat = self.proj[name](feat)
            feats.append(feat)

        stacked = torch.stack(feats, dim=1)               # [B, 3, proj_dim]
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        fused = attn_out.mean(dim=1)                       # [B, proj_dim]
        consensus_feat = self.fusion_mlp(fused)            # [B, 128]

        pred = self.prediction_head(consensus_feat)        # [B, 7]
        return pred, consensus_feat

class ConsensusModel_Scratch(nn.Module):
    """
    消融实验 D：从零训练，特征提取器随机初始化且参与训练。
    """
    def __init__(self, config, feature_dim=128, proj_dim=64, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = ['era5', 'colm', 'smci']   # 固定产品名

        # 三个可训练的特征提取器（注意 input_dim 需与数据一致）
        self.extractors = nn.ModuleDict({
            name: TemporalFeatureExtractor(input_dim=12)
            for name in self.product_names
        })

        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim) for name in self.product_names
        })

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.prediction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            ) for name in self.product_names
        })

    def forward(self, dynamic_input, static_input):
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)  # [B, feature_dim]
            feat = self.proj[name](feat)
            feats.append(feat)

        stacked = torch.stack(feats, dim=1)
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        fused = attn_out.mean(dim=1)
        consensus_feat = self.fusion_mlp(fused)

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat

class ConsensusModel_NoProj(nn.Module):
    """
    消融实验 E：移除投影层，使用原始特征维度进行注意力。
    """
    def __init__(self, config, pretrained_paths, feature_dim=128, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())

        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        # 无投影层，直接使用 feature_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 融合 MLP 输入维度为 feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        self.prediction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            ) for name in self.product_names
        })

    def forward(self, dynamic_input, static_input):
        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)  # [B, feature_dim]
            feats.append(feat)

        stacked = torch.stack(feats, dim=1)          # [B, 3, feature_dim]
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        fused = attn_out.mean(dim=1)                  # [B, feature_dim]

        consensus_feat = self.fusion_mlp(fused)       # [B, 128]

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat