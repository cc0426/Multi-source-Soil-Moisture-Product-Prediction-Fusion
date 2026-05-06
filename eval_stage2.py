import numpy as np
import torch
import xarray as xr
import pandas as pd
from config import DataConfig, TrainingConfig, ModelConfig, PRODUCT_CONFIGS
from model import ConsensusModel  # 导入阶段二的共识模型

# -------------------- 数据加载 --------------------
def get_data(data_config):
    """加载并归一化动态、静态数据及产品标签（与之前相同）"""
    lon = np.load('./dataset/forcing_lon.npy')
    lat = np.load('./dataset/forcing_lat.npy')
    station_mask = np.load('./dataset/station_masks.npy', allow_pickle=True).item()
    params = np.load('./dataset/normalization_params.npy', allow_pickle=True).item()
    dynamic_means = params['dynamic_means']
    dynamic_stds = params['dynamic_stds']
    static_means = params['static_means']
    static_stds = params['static_stds']

    start_date = data_config.test_start_date
    end_date = data_config.test_end_date

    start_date_obj = pd.to_datetime(start_date)
    previous_year_start = (start_date_obj - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    dynamic_start_date = previous_year_start
    dynamic_time_index = pd.date_range(start=dynamic_start_date, end=end_date, freq='D')
    dynamic_data_time_index = pd.date_range(start=data_config.data_start_date, end=data_config.data_end_date, freq='D')
    dynamic_time_mask = dynamic_data_time_index.isin(dynamic_time_index)

    time_index = pd.date_range(start=start_date, end=end_date, freq='D')
    data_time_index = pd.date_range(start=data_config.data_start_date, end=data_config.data_end_date, freq='D')
    time_mask = data_time_index.isin(time_index)

    # 动态数据
    dynamic_values = np.load(data_config.dynamic_data_path)  # (全部时间, 128, 128, 7)
    dynamic_data = dynamic_values[dynamic_time_mask]        # (测试期+前一年, 128, 128, 7)

    # 静态数据
    static_variables = np.load(data_config.static_data_path)  # (128, 128, 5)

    # 产品数据
    product_data = {}
    for product_name in PRODUCT_CONFIGS.keys():
        product_values = np.load(PRODUCT_CONFIGS[product_name]['filepath'])
        extract_data = product_values[time_mask]              # (测试期, 128, 128)
        product_data[product_name] = extract_data

    # 归一化
    dynamic_tensor = (dynamic_data - dynamic_means) / (dynamic_stds - dynamic_means)
    static_tensor = np.full_like(static_variables, np.nan)
    for i, key in enumerate(['clay_05cm', 'sand_05cm', 'silt_05cm', 'DEM', 'landcover']):
        static_tensor[:, :, i] = (static_variables[:, :, i] - static_means[key]) / (static_stds[key] - static_means[key])

    print('数据加载与归一化完成')
    return dynamic_tensor, static_tensor, product_data

# -------------------- 样本构建函数 --------------------
def prepare_samples_for_point(dynamic_series, static_vec):
    """
    为单个网格点构建LSTM输入样本（不依赖特定产品标签）
    dynamic_series: (T_dynamic, n_feat) 该点的动态序列（包含前一年）
    static_vec: (n_static) 静态特征
    返回:
        x: (n_samples, 365, n_feat)
        static: (n_samples, n_static)
        n_samples: 可生成的样本数
    """
    total_timesteps, n_feat = dynamic_series.shape
    # 假设标签长度 = T_dynamic - 365
    label_timesteps = total_timesteps - 365
    n_samples = label_timesteps - 7 + 1
    x = np.zeros((n_samples, 365, n_feat), dtype=np.float32)
    static = np.tile(static_vec, (n_samples, 1)).astype(np.float32)

    for i in range(n_samples):
        x[i] = dynamic_series[i:i+365]          # 前365天作为输入
    return x, static, n_samples


# -------------------- 主评估函数 --------------------
def main():
    data_config = DataConfig()
    training_config = TrainingConfig()
    device = training_config.device

    # 模型参数
    model_config = ModelConfig()

    # 加载数据
    dynamic_tensor, static_tensor, product_data = get_data(data_config)
    T_dyn, lat_len, lon_len, n_feat = dynamic_tensor.shape
    # 测试期标签长度（从product_data中获取，用于验证）
    T_label = product_data['era5'].shape[0]
    n_samples = T_label - 7 + 1           # 每个点的样本数

    # 初始化存储预测结果的字典
    predictions = {}
    observations = {}

    # 产品名称（必须与阶段二模型中的顺序一致）
    products = ['era5', 'smci', 'colm']

    # 加载阶段二的共识模型
    print("\n========== 加载阶段二共识模型 ==========")
    # 预训练模型路径（用于构建FrozenFeatureExtractor，但阶段二模型已包含）
    # 注意：这里需要提供预训练模型的路径，因为ConsensusModel需要它们来初始化特征提取器
    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }

    # 创建共识模型（与训练时相同的参数）
    consensus_model = ConsensusModel(
        config=training_config,
        pretrained_paths=pretrained_paths,
        feature_dim=128,      # 应与ProductModel的shared_dim一致
        proj_dim=64,
        num_heads=4
    )

    # 加载训练好的权重
    checkpoint_path = './checkpoints/stage1/stage2/stage2_best_model.pth'  # 请根据实际路径修改
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    consensus_model.load_state_dict(checkpoint['model_state_dict'])
    consensus_model = consensus_model.to(device)
    consensus_model.eval()
    print("模型加载完成")

    # 初始化每个产品的预测数组
    for prod in products:
        predictions[prod] = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)
        observations[prod] = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)

    # 逐网格点预测
    for i in range(lat_len):
        print(f"  处理纬度 {i+1}/{lat_len}")
        for j in range(lon_len):
            # 获取该点的动态序列、静态向量
            dyn_series = dynamic_tensor[:, i, j, :]          # (T_dyn, n_feat)
            static_vec = static_tensor[i, j, :]              # (static_dim)

            # 构建模型输入样本
            x, static, n_pts = prepare_samples_for_point(dyn_series, static_vec)
            if n_pts == 0:
                continue

            # 转换为 tensor
            x_t = torch.from_numpy(x).float().to(device)
            static_t = torch.from_numpy(static).float().to(device)

            with torch.no_grad():
                preds_dict, _ = consensus_model(x_t, static_t)   # 返回 (preds, consensus_feat)
                # preds_dict 是一个字典，键为产品名，值为 [n_pts, 7]

            # 存储每个产品的预测
            for prod in products:
                pred = preds_dict[prod].cpu().numpy()  # [n_pts, 7]
                predictions[prod][:n_pts, i, j, :] = pred
                # 观测值直接从 product_data 中提取（注意 product_data 的形状为 (T_label, lat, lon)）
                # 需要将观测值整理成 (n_pts, 7) 的形式
                prod_series = product_data[prod][:, i, j]  # (T_label)
                for k in range(n_pts):
                    observations[prod][k, i, j, :] = prod_series[k:k+7]

    # 保存预测结果
    for prod in products:
        np.save(f'./eval_data/pred_stage2_{prod}.npy', predictions[prod])
        np.save(f'./eval_data/obs_stage2_{prod}.npy', observations[prod])
        print(f"{prod} 预测结果已保存到 ./eval_data/pred_stage2_{prod}.npy")

    print("\n所有预测完成！")


if __name__ == "__main__":
    main()