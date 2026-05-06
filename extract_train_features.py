import torch
import numpy as np
import os
from config import DataConfig, TrainingConfig, ModelConfig
from data_loader import create_data_loaders
from model import ProductModel,ConsensusModel


def extract_train_features(n_samples=1000000, batch_size=512, save_dir='./eval_data_1'):
    """
    从训练集中随机采样 n_samples 个样本，提取阶段一和阶段二特征，并保存标签（第1天）。
    """
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    data_config = DataConfig()
    training_config = TrainingConfig()
    model_config = ModelConfig()
    training_config.device = device

    # 创建训练数据加载器（shuffle=True 以确保随机采样）
    train_loader, _, _, _ = create_data_loaders(
        data_config,
        training_config=training_config,
        grid_mask_path='./dataset/mask_Northeast_China.npy',
        normalize=True
    )
    print(f"训练集总样本数: {len(train_loader.dataset)}")

    # 加载阶段一模型
    products = ['era5', 'colm', 'smci']
    feat_dim_stage1 = model_config.shared_dim
    stage1_models = {}
    for prod in products:
        checkpoint_path = f'./checkpoints/stage1/{prod}/stage1_best_model.pth'
        model = ProductModel(input_dim=model_config.input_dim,
                             shared_dim=feat_dim_stage1,
                             output_days=7)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        stage1_models[prod] = model
        print(f"加载阶段一模型 {prod} 完成")

    # 加载阶段二共识模型
    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }
    consensus_model = ConsensusModel(
        config=training_config,
        pretrained_paths=pretrained_paths,
        feature_dim=feat_dim_stage1,
        proj_dim=64,
        num_heads=4
    )
    checkpoint_stage2 = torch.load('./checkpoints/stage1/stage2/stage2_best_model.pth', map_location=device, weights_only=False)
    consensus_model.load_state_dict(checkpoint_stage2['model_state_dict'])
    consensus_model = consensus_model.to(device)
    consensus_model.eval()
    print("加载阶段二共识模型完成")

    # 准备存储列表
    train_feat_stage1 = {prod: [] for prod in products}
    train_feat_stage2 = []
    train_obs = {prod: [] for prod in products}

    collected = 0
    for batch_idx, batch in enumerate(train_loader):
        if collected >= n_samples:
            break

        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)

        with torch.no_grad():
            # 阶段一特征
            for prod in products:
                _, feats = stage1_models[prod](dynamic, static)
                train_feat_stage1[prod].append(feats.cpu().numpy())

            # 阶段二共识特征
            _, consensus_feat = consensus_model(dynamic, static)
            train_feat_stage2.append(consensus_feat.cpu().numpy())

        # 标签（取第1天）
        for prod in products:
            target_list = batch['product_targets'][prod]
            target_tensor = torch.stack(target_list, dim=0).to(device)
            train_obs[prod].append(target_tensor[:, 0].cpu().numpy())  # 只取第1天

        collected += dynamic.size(0)
        if batch_idx % 10 == 0:
            print(f"已收集 {collected} 个样本")

    # 合并列表并截取所需数量
    for prod in products:
        train_feat_stage1[prod] = np.concatenate(train_feat_stage1[prod], axis=0)[:n_samples]
        train_obs[prod] = np.concatenate(train_obs[prod], axis=0)[:n_samples]
    train_feat_stage2 = np.concatenate(train_feat_stage2, axis=0)[:n_samples]

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'train_feat_stage1_era5.npy'), train_feat_stage1['era5'])
    np.save(os.path.join(save_dir, 'train_feat_stage1_colm.npy'), train_feat_stage1['colm'])
    np.save(os.path.join(save_dir, 'train_feat_stage1_smci.npy'), train_feat_stage1['smci'])
    np.save(os.path.join(save_dir, 'train_feat_stage2_consensus.npy'), train_feat_stage2)
    for prod in products:
        np.save(os.path.join(save_dir, f'train_obs_{prod}_day1.npy'), train_obs[prod])
    print(f"训练集特征保存完成，共 {collected} 个样本")

if __name__ == '__main__':
    extract_train_features(n_samples=1000000)  # 可根据需要调整采样数