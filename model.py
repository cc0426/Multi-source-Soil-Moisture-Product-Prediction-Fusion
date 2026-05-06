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


    def forward(self, x_time, x_static):
        batch_size, seq_len, _ = x_time.shape
        static_expanded = x_static.unsqueeze(1).expand(-1, seq_len, -1)  # [512, 365, 5]

        x = torch.cat([x_time, static_expanded], dim=-1)  # [512, 365, 12]


        temporal_features,_ = self.encoder(x)  # [batch, 365, d_model]
        features = temporal_features[:,-1,:]

        return features

class ProductModel(nn.Module):
    def __init__(self, input_dim, shared_dim, output_days=7):
        super().__init__()
        self.temporal_extractor = TemporalFeatureExtractor(input_dim=input_dim)
        self.predictor = nn.Linear(shared_dim, output_days) 

    def forward(self, dynamic_input, static_input):
        features = self.temporal_extractor(dynamic_input, static_input)  # [B, shared_dim]

        pred = self.predictor(features)  # [B, 7]

        return pred, features



class FrozenFeatureExtractor(nn.Module):

    def __init__(self, product_model_path, device, feature_dim=128):
        super().__init__()

        checkpoint = torch.load(product_model_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  


        self.extractor = ProductModel(input_dim=12, shared_dim=feature_dim, output_days=7)
        self.extractor.load_state_dict(state_dict, strict=False) 

        for param in self.extractor.parameters():
            param.requires_grad = False
        self.extractor.eval() 

    def forward(self, dynamic_input, static_input):

        _, features = self.extractor(dynamic_input, static_input)
        return features


class ConsensusModel(nn.Module):

    def __init__(self, config, pretrained_paths, feature_dim=128, proj_dim=64, num_heads=4):
       
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())


        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })

        self.proj = nn.ModuleDict({
            name: nn.Linear(feature_dim, proj_dim)
            for name in self.product_names
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

        self.use_diversity_loss = getattr(config, 'use_diversity_loss', False)

    def forward(self, dynamic_input, static_input, return_attention=False):

        feats = []
        for name in self.product_names:
            feat = self.extractors[name](dynamic_input, static_input)  # [B, feature_dim]
            feat = self.proj[name](feat)                               # [B, proj_dim]
            feats.append(feat)


        stacked = torch.stack(feats, dim=1)


        attn_out, attn_weights = self.cross_attention(stacked, stacked, stacked,average_attn_weights=False)  # [B, 3, proj_dim]


        fused = attn_out.mean(dim=1)  # [B, proj_dim]


        consensus_feat = self.fusion_mlp(fused)  # [B, 128]


        preds = {}
        for name in self.product_names:
            preds[name] = self.prediction_heads[name](consensus_feat)  # [B, 7]

        if return_attention:
            return preds, consensus_feat, attn_weights
        return preds, consensus_feat

    def get_diversity_loss(self, product_features):

        if len(product_features) < 2:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        n_pairs = 0
        for i in range(len(product_features)):
            for j in range(i+1, len(product_features)):
                sim = F.cosine_similarity(product_features[i], product_features[j], dim=-1).mean()
                loss += sim
                n_pairs += 1

        return loss / n_pairs

class ConsensusModel_MeanFusion(nn.Module):

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
            feat = self.proj[name](feat)                               # [B, proj_dim]
            feats.append(feat)


        stacked = torch.stack(feats, dim=1)   # [B, 3, proj_dim]
        fused = stacked.mean(dim=1)           # [B, proj_dim]

        consensus_feat = self.fusion_mlp(fused)  # [B, 128]

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat

class ConsensusModel_ConcatFusion(nn.Module):

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


        fused = torch.cat(feats, dim=-1)      # [B, proj_dim * 3]

        consensus_feat = self.fusion_mlp(fused)  # [B, 128]

        preds = {name: head(consensus_feat) for name, head in self.prediction_heads.items()}
        return preds, consensus_feat

class ConsensusModel_SingleTask(nn.Module):

    def __init__(self, config, pretrained_paths, target_name, feature_dim=128, proj_dim=64, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())
        self.target = target_name     

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

    def __init__(self, config, feature_dim=128, proj_dim=64, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = ['era5', 'colm', 'smci'] 

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

    def __init__(self, config, pretrained_paths, feature_dim=128, num_heads=4):
        super().__init__()
        self.device = config.device
        self.product_names = list(pretrained_paths.keys())

        self.extractors = nn.ModuleDict({
            name: FrozenFeatureExtractor(path, self.device, feature_dim)
            for name, path in pretrained_paths.items()
        })


        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )


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
