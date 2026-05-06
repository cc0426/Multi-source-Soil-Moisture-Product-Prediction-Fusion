import numpy as np
import torch
import xarray as xr
import pandas as pd
from config import DataConfig, TrainingConfig, ModelConfig, PRODUCT_CONFIGS
from model import ProductModel


# -------------------- 数据加载 --------------------
def get_data(data_config):
    """加载并归一化动态、静态数据及产品标签"""
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
def prepare_samples_for_point(dynamic_series, static_vec, product_series):
    """
    为单个网格点构建LSTM输入样本和标签
    dynamic_series: (T_dynamic, n_feat) 该点的动态序列（包含前一年）
    static_vec: (n_static) 静态特征
    product_series: (T_label) 该点的产品标签序列（测试期）
    返回:
        x: (n_samples, 365, n_feat)
        y: (n_samples, 7)
        static: (n_samples, n_static)
    """
    total_timesteps, n_feat = dynamic_series.shape
    label_timesteps = len(product_series)
    # 确保动态序列长度足够（动态长度 = 标签长度 + 365）
    assert total_timesteps >= label_timesteps + 365, "动态数据长度不足"
    n = label_timesteps - 7 + 1
    x = np.zeros((n, 365, n_feat), dtype=np.float32)
    y = np.zeros((n, 7), dtype=np.float32)
    static = np.tile(static_vec, (n, 1)).astype(np.float32)

    for i in range(n):
        x[i] = dynamic_series[i:i+365]          # 前365天作为输入
        y[i] = product_series[i:i+7]             # 后7天作为标签
    return x, y, static


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
    # 测试期标签长度（从product_data中获取）
    T_label = product_data['era5'].shape[0]
    n_samples = T_label - 7 + 1           # 每个点的样本数

    # 存储所有模型的预测和标签
    predictions = {}
    observations = {}

    # 三个产品名称
    products = ['era5', 'smci', 'colm']

    for prod in products:
        print(f"\n========== 处理产品 {prod} 的模型 ==========")
        # 加载对应模型
        checkpoint_path = f'./checkpoints/stage1/{prod}/stage1_best_model.pth'  # 根据实际路径调整
        model = ProductModel(
            input_dim=model_config.input_dim,  # 动态特征维度
            shared_dim=model_config.shared_dim,  # 时序特征提取器输出的维度
            output_days=7
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # 关键：将模型移动到设备
        model.eval()
        if model is None:
            print("模型加载失败，跳过")
            continue

        # 初始化该模型的预测和标签数组
        pred_arr = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)
        obs_arr = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)

        # 逐网格点预测
        for i in range(lat_len):
            print(f"  处理纬度 {i+1}/{lat_len}")
            for j in range(lon_len):
                # 获取该点的动态序列、静态向量和产品标签
                dyn_series = dynamic_tensor[:, i, j, :]          # (T_dyn, n_feat)
                static_vec = static_tensor[i, j, :]              # (static_dim)
                prod_series = product_data[prod][:, i, j]        # (T_label)

                # 构建样本
                x, y, static = prepare_samples_for_point(dyn_series, static_vec, prod_series)

                # 转换为 tensor
                x_t = torch.from_numpy(x).float().to(device)
                static_t = torch.from_numpy(static).float().to(device)

                with torch.no_grad():
                    out,_ = model(x_t, static_t)   # 模型输出形状 (n_samples, 7)

                # 保存预测和标签
                pred_arr[:, i, j, :] = out.cpu().numpy()
                obs_arr[:, i, j, :] = y

        # 存储
        predictions[prod] = pred_arr
        observations[prod] = obs_arr

        # 保存到文件
        np.save(f'./eval_data/pred_{prod}.npy', pred_arr)
        np.save(f'./eval_data/obs_{prod}.npy', obs_arr)
        print(f"{prod} 预测结果已保存")



if __name__ == "__main__":
    main()