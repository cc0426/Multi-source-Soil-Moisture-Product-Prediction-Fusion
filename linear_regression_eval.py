"""
predict_linear_day1.py

"""

import numpy as np
import torch
import pandas as pd
import os
import joblib
from config import DataConfig, TrainingConfig, ModelConfig, PRODUCT_CONFIGS
from model import ProductModel, ConsensusModel


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
    if n_samples <= 0:
        return None, None, 0
    x = np.zeros((n_samples, 365, n_feat), dtype=np.float32)
    static = np.tile(static_vec, (n_samples, 1)).astype(np.float32)
    for i in range(n_samples):
        x[i] = dynamic_series[i:i+365]
    return x, static, n_samples

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    data_config = DataConfig()
    training_config = TrainingConfig()
    model_config = ModelConfig()

    dynamic_tensor, static_tensor, product_data = get_data(data_config)
    T_dyn, lat_len, lon_len, n_feat = dynamic_tensor.shape
    T_label = product_data['era5'].shape[0]
    n_samples = T_label - 7 + 1

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



    china_mask = np.load('/home/zhangcheng/Soil_Moisture/CML_FD/dataset/mask_Northeast_China.npy')
    print(f"掩码形状: {china_mask.shape}")


    targets = ['ERA5', 'CoLM', 'SMCI']  
    feature_sources = {
        'Stage1_ERA5': stage1_models['era5'],
        'Stage1_CoLM': stage1_models['colm'],
        'Stage1_SMCI': stage1_models['smci'],
        'Stage2_Consensus': consensus_model,
    }

    pred_arrays = {} 

    obs_arrays = {} 

    for target in targets:
        target_lower = target.lower()

        obs_arrays[target] = np.full((n_samples, lat_len, lon_len), np.nan, dtype=np.float32)

        pred_arrays[target] = {}


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

                feats_stage1 = {}
                for prod in products:
                    _, feats = stage1_models[prod](x_t, static_t)
                    feats_stage1[prod] = feats.cpu().numpy()  # (n_pts, feat_dim)


                _, consensus_feat = consensus_model(x_t, static_t)
                consensus_feat_np = consensus_feat.cpu().numpy()  # (n_pts, 128)


            for target in targets:
                target_lower = target.lower()

                prod_series = product_data[target_lower][:, i, j]

                for k in range(n_pts):
                    obs_arrays[target][k, i, j] = prod_series[k]


            for target in targets:
                target_lower = target.lower()
                for feat_name in feature_sources.keys():

                    if feat_name == 'Stage2_Consensus':
                        feat_np = consensus_feat_np
                    else:
                        prod_name = feat_name.split('_')[1].lower()
                        feat_np = feats_stage1[prod_name]


                    model_path = f"./linear_models/ridge_{target}_{feat_name}.pkl"
                    if not os.path.exists(model_path):
                        
                        continue
                    lin_model = joblib.load(model_path)


                    valid_mask = ~np.isnan(feat_np).any(axis=1) 
                    if valid_mask.any():
                        pred_valid = lin_model.predict(feat_np[valid_mask])

                        pred_full = np.full(n_pts, np.nan, dtype=np.float32)
                        pred_full[valid_mask] = pred_valid
                    else:
                        pred_full = np.full(n_pts, np.nan, dtype=np.float32)


                    if feat_name not in pred_arrays[target]:

                        pred_arr = np.full((n_samples, lat_len, lon_len), np.nan, dtype=np.float32)
                        pred_arrays[target][feat_name] = pred_arr
                    else:
                        pred_arr = pred_arrays[target][feat_name]


                    pred_arr[:n_pts, i, j] = pred_full


    output_dir = "./linear_predictions_day1"
    os.makedirs(output_dir, exist_ok=True)


    for target in targets:
        target_lower = target.lower()
        obs_arr = obs_arrays[target]
        obs_filename = f"obs_{target_lower}.npy"
        obs_path = os.path.join(output_dir, obs_filename)
        np.save(obs_path, obs_arr)

    for target in targets:
        target_lower = target.lower()
        for feat_name, pred_arr in pred_arrays[target].items():
            filename = f"pred_linear_{target_lower}_from_{feat_name}.npy"
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, pred_arr)


if __name__ == "__main__":
    main()
