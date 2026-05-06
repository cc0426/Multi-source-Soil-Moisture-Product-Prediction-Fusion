import numpy as np
import torch
import xarray as xr
import pandas as pd
from config import DataConfig, TrainingConfig, ModelConfig, PRODUCT_CONFIGS
from model import ProductModel



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


    static_variables = np.load(data_config.static_data_path)  # (128, 128, 5)


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


def prepare_samples_for_point(dynamic_series, static_vec, product_series):

    total_timesteps, n_feat = dynamic_series.shape
    label_timesteps = len(product_series)

    n = label_timesteps - 7 + 1
    x = np.zeros((n, 365, n_feat), dtype=np.float32)
    y = np.zeros((n, 7), dtype=np.float32)
    static = np.tile(static_vec, (n, 1)).astype(np.float32)

    for i in range(n):
        x[i] = dynamic_series[i:i+365]    
        y[i] = product_series[i:i+7]        
    return x, y, static



def main():
    data_config = DataConfig()
    training_config = TrainingConfig()
    device = training_config.device


    model_config = ModelConfig()

    dynamic_tensor, static_tensor, product_data = get_data(data_config)
    T_dyn, lat_len, lon_len, n_feat = dynamic_tensor.shape

    T_label = product_data['era5'].shape[0]
    n_samples = T_label - 7 + 1      


    predictions = {}
    observations = {}


    products = ['era5', 'smci', 'colm']

    for prod in products:

        checkpoint_path = f'./checkpoints/stage1/{prod}/stage1_best_model.pth'
        model = ProductModel(
            input_dim=model_config.input_dim,  
            shared_dim=model_config.shared_dim, 
            output_days=7
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device) 
        model.eval()
        if model is None:

            continue


        pred_arr = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)
        obs_arr = np.full((n_samples, lat_len, lon_len, 7), np.nan, dtype=np.float32)


        for i in range(lat_len):

            for j in range(lon_len):

                dyn_series = dynamic_tensor[:, i, j, :]          # (T_dyn, n_feat)
                static_vec = static_tensor[i, j, :]              # (static_dim)
                prod_series = product_data[prod][:, i, j]        # (T_label)


                x, y, static = prepare_samples_for_point(dyn_series, static_vec, prod_series)


                x_t = torch.from_numpy(x).float().to(device)
                static_t = torch.from_numpy(static).float().to(device)

                with torch.no_grad():
                    out,_ = model(x_t, static_t)   #(n_samples, 7)


                pred_arr[:, i, j, :] = out.cpu().numpy()
                obs_arr[:, i, j, :] = y


        predictions[prod] = pred_arr
        observations[prod] = obs_arr


        np.save(f'./eval_data/pred_{prod}.npy', pred_arr)
        np.save(f'./eval_data/obs_{prod}.npy', obs_arr)




if __name__ == "__main__":
    main()
