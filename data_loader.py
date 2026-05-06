from torch.utils.data import DataLoader
import os
import xarray as xr
from pathlib import Path
from config import DataConfig, PRODUCT_CONFIGS, TrainingConfig
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict


def zero_fill_and_mask(tensor_np):
    """
    tensor_np: 可能含 NaN 的 numpy 数组
    返回:
        filled_tensor:  NaN→0 的 float32 torch tensor
        mask_tensor:    1=有效 0=缺失, 同形状
    """
    tensor_np = np.asarray(tensor_np, dtype=np.float32)
    mask = ~np.isnan(tensor_np)  # bool
    filled = np.nan_to_num(tensor_np, nan=0.0)  # 0 填充
    return torch.from_numpy(filled), torch.from_numpy(mask.astype(np.float32))


class SoilMoistureDataset(Dataset):
    """土壤湿度数据集 - 支持训练/验证/测试不同模式"""

    def __init__(self, config, mode='train',
                 n_iter_per_epoch: Optional[int] = None,
                 batch_size: int = 32,
                 normalize: bool = True,
                 norm_params: Optional[Dict] = None,
                 grid_mask_path: Optional[str] = None,
                 test_full_region: bool = True,
                 include_previous_year: bool = False,
                 seed: int = 42):
        """
        Args:
            mode: 'train', 'val', 'test'
            n_iter_per_epoch: 训练和验证时每个epoch的迭代次数
            batch_size: 每个batch的样本数
            test_full_region: 测试时是否使用整个区域（忽略mask）
        """
        self.config = config
        self.mode = mode
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.test_full_region = test_full_region
        self.grid_mask_path = grid_mask_path
        self._set_seed()
        self.include_previous_year = include_previous_year
        self.seq_len = 365
        self.forecast_horizon = 7

        self.data_loader = RealDataLoader(config)
        self._set_date_range()
        self._load_data()
        self._process_grid_points()
        self._setup_normalization(norm_params)
        self._precompute_static_features()
        self._setup_sampling_strategy(n_iter_per_epoch)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _set_date_range(self):
        if self.mode == 'train':
            self.start_date = self.config.train_start_date
            self.end_date = self.config.train_end_date
        elif self.mode == 'val':
            self.start_date = self.config.val_start_date
            self.end_date = self.config.val_end_date
        else:
            self.start_date = self.config.test_start_date
            self.end_date = self.config.test_end_date

        if self.include_previous_year and self.mode in ['val', 'test']:
            start_date_obj = pd.to_datetime(self.start_date)
            previous_year_start = (start_date_obj - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            self.dynamic_start_date = previous_year_start
            print(f"{self.mode}模式：包含前一年数据，动态特征从 {self.dynamic_start_date} 开始")
        else:
            self.dynamic_start_date = self.start_date

    def _load_data(self):
        print(f"加载{self.mode}数据:")
        print(f"  动态特征: {self.dynamic_start_date} 到 {self.end_date}")
        print(f"  标签数据: {self.start_date} 到 {self.end_date}")

        self.dynamic_data = self.data_loader.load_dynamic_data(self.dynamic_start_date, self.end_date)
        self.static_data = self.data_loader.load_static_data()
        self.product_data = {}
        for product_name in PRODUCT_CONFIGS.keys():
            self.product_data[product_name] = self.data_loader.load_product_data(
                product_name, self.start_date, self.end_date
            )
        self.station_data = self.data_loader.load_station_data(self.start_date, self.end_date)

        self.lat_size = self.dynamic_data.sizes['lat']
        self.lon_size = self.dynamic_data.sizes['lon']
        self.total_timesteps = len(self.dynamic_data.time)
        self.label_timesteps = len(self.product_data[list(self.product_data.keys())[0]].time)

        print(f"网格尺寸: {self.lat_size}×{self.lon_size}")
        print(f"动态数据时间步: {self.total_timesteps} (包含前一年: {self.include_previous_year})")
        print(f"标签数据时间步: {self.label_timesteps}")

    def _process_grid_points(self):
        if self.mode == 'test' and self.test_full_region:
            self.valid_grid_indices = []
            for i in range(self.lat_size):
                for j in range(self.lon_size):
                    self.valid_grid_indices.append((i, j))
            print(f"测试模式：使用整个区域的所有{len(self.valid_grid_indices)}个格点")
        else:
            self.grid_mask = self._load_grid_mask(self.grid_mask_path)
            self.valid_grid_indices = self._get_valid_grid_indices()
            print(f"{self.mode}模式：使用{len(self.valid_grid_indices)}个有效格点")

    def _load_grid_mask(self, grid_mask_path):
        if grid_mask_path is None:
            print(f"警告：{self.mode}模式未提供grid_mask_path")
            return None
        mask = np.load(grid_mask_path)
        print(f"加载格点mask，形状: {mask.shape}")
        print(f"有效格点比例: {mask.sum() / mask.size:.2%}")
        return mask

    def _get_valid_grid_indices(self):
        if self.grid_mask is None:
            valid_indices = []
            for i in range(self.lat_size):
                for j in range(self.lon_size):
                    valid_indices.append((i, j))
        else:
            valid_indices = []
            for i in range(self.grid_mask.shape[0]):
                for j in range(self.grid_mask.shape[1]):
                    if self.grid_mask[i, j] == 1:
                        valid_indices.append((i, j))
        return valid_indices

    def _setup_normalization(self, norm_params):
        if self.normalize and self.mode == 'train' and norm_params is None:
            self._compute_normalization_params()
        elif self.normalize and norm_params is not None:
            self._set_normalization_params(norm_params)
        elif self.normalize and self.mode in ['val', 'test'] and norm_params is None:
            raise ValueError(f"{self.mode}模式必须提供norm_params参数")

    # ------------------ 修改点：格点级动态特征归一化参数计算 ------------------
    def _compute_normalization_params(self):
        """计算格点级归一化参数（动态特征按格点计算最小最大值，静态特征保持全局）"""
        print(f"计算训练数据的格点级归一化参数...")
        dynamic_values = self.dynamic_data.values  # [T, H, W, C]
        T, H, W, C = dynamic_values.shape

        # 初始化格点级最小/最大值数组
        dynamic_mins = np.full((H, W, C), np.nan, dtype=np.float32)
        dynamic_maxs = np.full((H, W, C), np.nan, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                for c in range(C):
                    feature_ts = dynamic_values[:, i, j, c]
                    valid_data = feature_ts[~np.isnan(feature_ts)]
                    if len(valid_data) > 0:
                        min_val = valid_data.min()
                        max_val = valid_data.max()
                        if max_val - min_val < 1e-6:
                            max_val = min_val + 1.0
                    else:
                        # 若全部缺失，设为 NaN（后续不会被采样）
                        min_val = np.nan
                        max_val = np.nan
                    dynamic_mins[i, j, c] = min_val
                    dynamic_maxs[i, j, c] = max_val

        self.dynamic_means = torch.from_numpy(dynamic_mins)  # [H, W, C]
        self.dynamic_stds = torch.from_numpy(dynamic_maxs)   # [H, W, C]

        # 静态特征保持全局归一化（按特征计算最小最大值）
        self.static_means, self.static_stds = {}, {}
        for key in ['clay_05cm', 'sand_05cm', 'silt_05cm', 'DEM', 'landcover']:
            static_values = self.static_data[key].values
            valid_values = static_values[~np.isnan(static_values)]
            if len(valid_values) > 0:
                static_min = float(valid_values.min())
                static_max = float(valid_values.max())
                if static_max - static_min < 1e-6:
                    static_max = static_min + 1.0
                self.static_means[key] = static_min
                self.static_stds[key] = static_max
            else:
                self.static_means[key] = 0.0
                self.static_stds[key] = 1.0

        print(f"训练数据格点级归一化参数计算完成")

    # ------------------ 修改点：设置格点级参数 ------------------
    def _set_normalization_params(self, norm_params):
        print(f"设置{self.mode}数据的归一化参数")
        self.dynamic_means = torch.tensor(norm_params['dynamic_means'], dtype=torch.float32)  # [H, W, C]
        self.dynamic_stds = torch.tensor(norm_params['dynamic_stds'], dtype=torch.float32)    # [H, W, C]
        self.static_means = norm_params['static_means']
        self.static_stds = norm_params['static_stds']

    def _precompute_static_features(self):
        print(f"预计算静态特征...")
        self.cached_static_features = {}
        for i in range(self.lat_size):
            for j in range(self.lon_size):
                static_tensor_list = []
                for key in ['clay_05cm', 'sand_05cm', 'silt_05cm', 'DEM', 'landcover']:
                    static_value = self.static_data[key].isel(lat=i, lon=j).values
                    if np.isnan(static_value):
                        value_tensor = torch.tensor(0.0).float()
                    else:
                        value_tensor = torch.tensor(static_value).float()
                    static_tensor_list.append(value_tensor)
                static_tensor = torch.stack(static_tensor_list, dim=0)
                self.cached_static_features[(i, j)] = static_tensor
        print(f"已预计算{len(self.cached_static_features)}个格点的静态特征")

    def _setup_sampling_strategy(self, n_iter_per_epoch):
        if self.mode == 'train':
            self.time_windows = list(range(0, self.total_timesteps - self.seq_len - self.forecast_horizon + 1))
            self.sampling_strategy = 'random'
            self.n_iter_per_epoch = n_iter_per_epoch if n_iter_per_epoch else 1000
            self._generate_random_sampling_plan()
        elif self.mode == 'val':
            self.sampling_strategy = 'random'
            self.n_iter_per_epoch = n_iter_per_epoch if n_iter_per_epoch else 1000
            if self.include_previous_year:
                self.time_offset = self.total_timesteps - self.label_timesteps
                print(f"时间偏移: {self.time_offset} 天")
                label_time_windows = list(range(0, self.label_timesteps - self.forecast_horizon + 1))
                self.time_windows = [tw + self.time_offset - self.seq_len for tw in label_time_windows]
                print(f"验证集：生成 {len(self.time_windows)} 个时间窗口")
            else:
                self.time_windows = list(range(0, self.total_timesteps - self.seq_len - self.forecast_horizon + 1))
            self._generate_random_sampling_plan()
        else:
            self.sampling_strategy = 'systematic'
            if self.include_previous_year:
                self.time_offset = self.total_timesteps - self.label_timesteps
                print(f"时间偏移: {self.time_offset} 天")
                label_time_windows = list(range(0, self.label_timesteps - self.forecast_horizon + 1))
                self.time_windows = [tw + self.time_offset - self.seq_len for tw in label_time_windows]
                print(f"测试集：生成 {len(self.time_windows)} 个时间窗口")
            else:
                self.time_windows = list(range(0, self.total_timesteps - self.seq_len - self.forecast_horizon + 1))
            self._generate_test_indices()

    def _generate_random_sampling_plan(self):
        self.time_window_plan = np.random.choice(
            self.time_windows,
            size=self.n_iter_per_epoch,
            replace=True
        )
        self.grid_plan = []
        for i in range(self.n_iter_per_epoch):
            grid_indices = np.random.choice(
                len(self.valid_grid_indices),
                size=self.batch_size,
                replace=True
            )
            self.grid_plan.append(grid_indices)
        print(f"{self.mode}模式：生成{self.n_iter_per_epoch}个batch的随机采样计划")

    def _generate_test_indices(self):
        print(f"为测试集生成系统采样索引...")
        self.test_time_windows = self.time_windows
        self.test_indices = []
        for time_idx in self.test_time_windows:
            num_grids = len(self.valid_grid_indices)
            num_batches = (num_grids + self.batch_size - 1) // self.batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_grids)
                grid_indices = list(range(start_idx, end_idx))
                self.test_indices.append({
                    'time_idx': time_idx,
                    'grid_indices': grid_indices,
                    'batch_idx': batch_idx,
                    'total_batches': num_batches
                })
        print(f"测试模式：生成{len(self.test_indices)}个测试batch")

    def refresh_sampling_plan(self):
        if self.sampling_strategy == 'random':
            self._generate_random_sampling_plan()

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_indices)
        else:
            return self.n_iter_per_epoch

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self._get_test_sample(idx)
        else:
            return self._get_random_sample(idx)

    def _get_test_sample(self, idx):
        sample_info = self.test_indices[idx]
        time_idx = sample_info['time_idx']
        grid_indices = sample_info['grid_indices']
        actual_grid_indices = []
        for i in grid_indices:
            if i < len(self.valid_grid_indices):
                actual_grid_indices.append(self.valid_grid_indices[i])
            else:
                actual_grid_indices.append(self.valid_grid_indices[0])
        batch = self._build_batch(time_idx, actual_grid_indices)
        batch['time_window'] = torch.tensor([time_idx])
        batch['batch_info'] = torch.tensor([
            sample_info['batch_idx'],
            sample_info['total_batches']
        ])
        return batch

    def _get_random_sample(self, idx):
        time_idx = self.time_window_plan[idx]
        grid_indices = self.grid_plan[idx]
        actual_grid_indices = [self.valid_grid_indices[i] for i in grid_indices]
        batch = self._build_batch(time_idx, actual_grid_indices)
        batch['time_window'] = torch.tensor([time_idx])
        return batch

    # ------------------ 修改点：应用格点级归一化 ------------------
    def _build_batch(self, time_idx, grid_indices):
        batch_dynamics = []
        batch_statics = []
        batch_product_targets_dict = {product_name: [] for product_name in self.product_data.keys()}

        for lat_idx, lon_idx in grid_indices:
            # 动态特征
            dynamic_slice = self.dynamic_data.isel(
                time=slice(time_idx, time_idx + self.seq_len),
                lat=lat_idx,
                lon=lon_idx
            )
            dynamic_values = dynamic_slice.values
            filled, mask = zero_fill_and_mask(dynamic_values)
            dynamic_tensor = filled  # [seq_len, C]

            # 格点级动态特征归一化
            if self.normalize:
                # 取出该格点的归一化参数（假设该格点参数非 NaN，否则应处理）
                dyn_mean = self.dynamic_means[lat_idx, lon_idx, :]  # [C]
                dyn_std = self.dynamic_stds[lat_idx, lon_idx, :]    # [C]
                dynamic_tensor = (dynamic_tensor - dyn_mean) / (dyn_std -dyn_mean)

            # 静态特征（全局归一化）
            static_tensor = self.cached_static_features[(lat_idx, lon_idx)].clone()
            if self.normalize:
                for i, key in enumerate(['clay_05cm', 'sand_05cm', 'silt_05cm', 'DEM', 'landcover']):
                    static_tensor[i] = (static_tensor[i] - self.static_means[key]) / (self.static_stds[key] - self.static_means[key])

            # 目标时间索引（处理前一年偏移）
            if self.include_previous_year and self.mode in ['val', 'test']:
                dynamic_target_start = time_idx + self.seq_len
                label_target_start = dynamic_target_start - self.time_offset
                label_target_end = label_target_start + self.forecast_horizon
                target_start = label_target_start
                target_end = label_target_end
            else:
                target_start = time_idx + self.seq_len
                target_end = target_start + self.forecast_horizon

            # 产品目标
            for product_name in self.product_data.keys():
                product_slice = self.product_data[product_name].isel(
                    time=slice(target_start, target_end),
                    lat=lat_idx,
                    lon=lon_idx
                )
                product_tensor = torch.from_numpy(product_slice.values).float()
                if np.isnan(product_tensor).any():
                    product_tensor = torch.where(
                        torch.isnan(product_tensor),
                        torch.tensor(float('nan')),
                        product_tensor
                    )
                batch_product_targets_dict[product_name].append(product_tensor)

            batch_dynamics.append(dynamic_tensor)
            batch_statics.append(static_tensor)

        batch_dynamics = torch.stack(batch_dynamics, dim=0)
        batch_statics = torch.stack(batch_statics, dim=0)

        return {
            'dynamic_features': batch_dynamics,
            'static_features': batch_statics,
            'product_targets': batch_product_targets_dict,
        }

    def _build_batch_from_indices(self, time_idx, grid_indices):
        return self._build_batch(time_idx, grid_indices)

    # ------------------ 修改点：返回格点级参数 ------------------
    def get_normalization_params(self):
        if not hasattr(self, 'dynamic_means'):
            return None
        dynamic_means_np = self.dynamic_means.cpu().numpy() if isinstance(self.dynamic_means, torch.Tensor) else self.dynamic_means
        dynamic_stds_np = self.dynamic_stds.cpu().numpy() if isinstance(self.dynamic_stds, torch.Tensor) else self.dynamic_stds
        static_means_np = {}
        static_stds_np = {}
        for key in self.static_means:
            if isinstance(self.static_means[key], torch.Tensor):
                static_means_np[key] = self.static_means[key].cpu().numpy()
                static_stds_np[key] = self.static_stds[key].cpu().numpy()
            else:
                static_means_np[key] = self.static_means[key]
                static_stds_np[key] = self.static_stds[key]
        return {
            'dynamic_means': dynamic_means_np,
            'dynamic_stds': dynamic_stds_np,
            'static_means': static_means_np,
            'static_stds': static_stds_np
        }

    def get_test_time_windows(self):
        if self.mode != 'test':
            return None
        return self.test_time_windows

    def get_test_grid_info(self):
        if self.mode != 'test':
            return None
        return {
            'grid_indices': self.valid_grid_indices,
            'lat_size': self.lat_size,
            'lon_size': self.lon_size
        }

class RealDataLoader:
    """真实数据加载器"""

    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.data_root)

    def load_data_with_range(self, data_path, start_date, end_date, time_dim='time'):
        """通用数据加载函数"""
        # 创建时间轴
        time_index = pd.date_range(start=start_date, end=end_date, freq='D')
        data_time_index = pd.date_range(start=self.config.data_start_date, end=self.config.data_end_date, freq='D')
        time_mask = data_time_index.isin(time_index)

        # 加载数据
        data_values = np.load(data_path)

        # 检查数据维度
        if len(data_values.shape) == 4:  # [时间, 纬度, 经度, 特征]
            extracted_data = data_values[time_mask]
        elif len(data_values.shape) == 3:  # [时间, 纬度, 经度]
            extracted_data = data_values[time_mask]
        else:
            raise ValueError(f"不支持的维度: {data_values.shape}")

        return extracted_data, time_index

    def load_dynamic_data(self, start_date, end_date):
        """加载动态气象数据"""
        print(f"加载动态数据: {start_date} 到 {end_date}")

        extracted_data, time_index = self.load_data_with_range(
            self.config.dynamic_data_path, start_date, end_date
        )

        lon = np.load('./dataset/forcing_lon.npy')
        lat = np.load('./dataset/forcing_lat.npy')

        # 创建xarray DataArray
        dynamic_data = xr.DataArray(
            extracted_data,
            dims=['time', 'lat', 'lon', 'feature'],
            coords={
                'time': time_index,
                'lat': lat,
                'lon': lon,
                'feature': ['surface_thermal_radiation', 'surface_solar_radiation_downwards_w_m2', 'specific_humidity',
                            'Precipitation_m_hr', '10m_u_component_of_wind', '10m_v_component_of_wind',
                            '2m_temperature']
            }
        )

        return dynamic_data

    def load_static_data(self):
        """加载静态数据"""
        static_variables = np.load(self.config.static_data_path)
        static_data = {
            'clay_05cm': xr.DataArray(
                static_variables[:, :, 0],
                dims=['lat', 'lon']
            ),
            'sand_05cm': xr.DataArray(
                static_variables[:, :, 1],
                dims=['lat', 'lon']
            ),
            'silt_05cm': xr.DataArray(
                static_variables[:, :, 2],
                dims=['lat', 'lon']
            ),
            'DEM': xr.DataArray(
                static_variables[:, :, 3],
                dims=['lat', 'lon']
            ),
            'landcover': xr.DataArray(
                static_variables[:, :, 4],
                dims=['lat', 'lon']
            ),
        }

        return static_data

    def load_product_data(self, product_name, start_date, end_date):
        """加载产品数据"""
        print(f"加载产品数据 {product_name}: {start_date} 到 {end_date}")

        extracted_data, time_index = self.load_data_with_range(
            PRODUCT_CONFIGS[product_name]['filepath'], start_date, end_date
        )

        lon = np.load('./dataset/forcing_lon.npy')
        lat = np.load('./dataset/forcing_lat.npy')

        product_data = xr.DataArray(
            extracted_data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': time_index,
                'lat': lat,
                'lon': lon
            }
        )

        return product_data

    def load_station_data(self, start_date, end_date):
        """加载站点数据（已转换为网格格式）"""
        print(f"加载站点数据: {start_date} 到 {end_date}")

        extracted_data, time_index = self.load_data_with_range(
            self.config.station_data_path, start_date, end_date
        )

        # 除以100进行缩放
        extracted_data = extracted_data

        lon = np.load('./dataset/forcing_lon.npy')
        lat = np.load('./dataset/forcing_lat.npy')

        station_data = xr.DataArray(
            extracted_data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': time_index,
                'lat': lat,
                'lon': lon
            }
        )

        return station_data


def create_data_loaders(data_config, training_config, grid_mask_path, normalize=True, save_norm_params=True):
    """创建数据加载器（验证集和测试集使用训练集的归一化参数）"""

    # 创建训练数据集
    print("创建训练数据集...")
    train_dataset = SoilMoistureDataset(data_config,
                                        mode='train',
                                        n_iter_per_epoch=200,
                                        batch_size=128,
                                        normalize=normalize,
                                        grid_mask_path=grid_mask_path,
                                        include_previous_year=False,
                                        seed=42)

    # 获取训练数据的归一化参数
    if normalize:
        norm_params = train_dataset.get_normalization_params()

        # 保存归一化参数到文件
        if save_norm_params and norm_params is not None:
            np.save('./dataset/normalization_params.npy', norm_params, allow_pickle=True)
            print("归一化参数已保存到 normalization_params.npy")
    else:
        norm_params = None

    # 创建验证数据集（使用训练集的归一化参数，随机采样）
    print("创建验证数据集...")
    val_dataset = SoilMoistureDataset(
        data_config,
        mode='val',
        n_iter_per_epoch=200,  # 验证集也用200个batch
        batch_size=128,
        normalize=True,
        norm_params=train_dataset.get_normalization_params(),
        grid_mask_path=grid_mask_path,
        include_previous_year=True,  # 验证集包含前一年数据
        seed=42)

    # 创建测试数据集（使用训练集的归一化参数，系统采样）
    print("创建测试数据集...")
    test_dataset = SoilMoistureDataset(
        data_config,
        mode='test',
        batch_size=1024,  # 测试集可以使用更大的batch_size
        normalize=True,
        norm_params=train_dataset.get_normalization_params(),
        test_full_region=True,  # 使用整个区域
        include_previous_year=True,  # 测试集包含前一年数据
        seed=42
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=training_config.num_workers,
        collate_fn=lambda x: x[0]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=training_config.num_workers,
        collate_fn=lambda x: x[0]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        # num_workers=training_config.num_workers,
        collate_fn=lambda x: x[0]
    )

    return train_loader, val_loader, test_loader, norm_params


def load_data_with_saved_params(data_config, training_config, norm_params_path='normalization_params.npy'):
    """使用已保存的归一化参数加载数据"""
    if os.path.exists(norm_params_path):
        norm_params = np.load(norm_params_path, allow_pickle=True).item()
        print(f"从 {norm_params_path} 加载归一化参数")
    else:
        print("警告：未找到归一化参数文件，将重新计算")
        norm_params = None

    return create_data_loaders(
        data_config,
        training_config,
        normalize=True,
        save_norm_params=False
    )