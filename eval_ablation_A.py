import numpy as np
import torch
import pandas as pd
from config import DataConfig, TrainingConfig, ModelConfig, PRODUCT_CONFIGS
from model import ConsensusModel_MeanFusion 


def get_data(data_config):
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

    dynamic_values = np.load(data_config.dynamic_data_path)
    dynamic_data = dynamic_values[dynamic_time_mask]

    static_variables = np.load(data_config.static_data_path)

    product_data = {}
    for product_name in PRODUCT_CONFIGS.keys():
        product_values = np.load(PRODUCT_CONFIGS[product_name]['filepath'])
        extract_data = product_values[time_mask]
        product_data[product_name] = extract_data

    dynamic_tensor = (dynamic_data - dynamic_means) / (dynamic_stds - dynamic_means)
    static_tensor = np.full_like(static_variables, np.nan)
    for i, key in enumerate(['clay_05cm', 'sand_05cm', 'silt_05cm', 'DEM', 'landcover']):
        static_tensor[:, :, i] = (static_variables[:, :, i] - static_means[key]) / (static_stds[key] - static_means[key])


    return dynamic_tensor, static_tensor, product_data

def prepare_samples_for_point(dynamic_series, static_vec):
    total_timesteps, n_feat = dynamic_series.shape
    label_timesteps = total_timesteps - 365
    n_samples = label_timesteps - 7 + 1
    x = np.zeros((n_samples, 365, n_feat), dtype=np.float32)
    static = np.tile(static_vec, (n_samples, 1)).astype(np.float32)
    for i in range(n_samples):
        x[i] = dynamic_series[i:i+365]
    return x, static, n_samples

def main():
    data_config = DataConfig()
    training_config = TrainingConfig()
    device = training_config.device
    model_config = ModelConfig()

    dynamic_tensor, static_tensor, product_data = get_data(data_config)
    T_dyn, lat_len, lon_len, n_feat = dynamic_tensor.shape
    T_label = product_data['era5'].shape[0]
    n_samples = T_label - 7 + 1

    products = ['era5', 'smci', 'colm']


    pretrained_paths = {
        'era5': './checkpoints/stage1/era5/stage1_best_model.pth',
        'colm': './checkpoints/stage1/colm/stage1_best_model.pth',
        'smci': './checkpoints/stage1/smci/stage1_best_model.pth'
    }
    model = ConsensusModel_MeanFusion(
        config=training_config,
        pretrained_paths=pretrained_paths,
        feature_dim=128,
        proj_dim=64
    )
    checkpoint_path = './checkpoints/stage2_ablation_A/stage2/stage2_best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()



    predictions = {prod: np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32) for prod in products}
    observations = {prod: np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32) for prod in products}


    for i in range(lat_len):

        for j in range(lon_len):
            dyn_series = dynamic_tensor[:, i, j, :]
            static_vec = static_tensor[i, j, :]
            x, static, n_pts = prepare_samples_for_point(dyn_series, static_vec)
            if n_pts == 0:
                continue

            x_t = torch.from_numpy(x).float().to(device)
            static_t = torch.from_numpy(static).float().to(device)

            with torch.no_grad():
                preds_dict, _ = model(x_t, static_t)  

            for prod in products:
                pred = preds_dict[prod].cpu().numpy()
                predictions[prod][:n_pts, i, j, :] = pred
                prod_series = product_data[prod][:, i, j]
                for k in range(n_pts):
                    observations[prod][k, i, j, :] = prod_series[k:k+7]


    for prod in products:
        np.save(f'./eval_data/pred_ablationA_{prod}.npy', predictions[prod])
        np.save(f'./eval_data/obs_ablationA_{prod}.npy', observations[prod])


if __name__ == "__main__":
    main()
