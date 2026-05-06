# config.py
import torch


class TrainingConfig:


    def __init__(self):
        self.num_epochs = 2000
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-3
        self.save_interval = 5 
        self.save_dir = "./checkpoints/stage1"


        self.optimizer_type = 'adam'  # 'adam', 'adamw', 'sgd'


        self.grad_clip = True
        self.grad_clip_value = 1.0

        self.early_stopping_threshold = 1e-4


        self.max_batches_per_epoch = None
        self.resume_checkpoint = "/home/zhangcheng/Soil_Moisture/CML_FD/checkpoints/stage1/stage1_best_model.pth" 


        self.use_wandb = False
        self.wandb_project = "soil_moisture_stage1"

        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class DataConfig:
 

    def __init__(self):
    
        self.data_root = 'data/'
        self.dynamic_data_path = './dataset/forcing.npy'
        self.static_data_path = './dataset/static_variables.npy'
        self.product_data_paths = {
            'ERA5-Land_soil': './dataset/ERA5_Land_SoilMoisture_2010-2018_SMAPgrid.npy',
            'SMCI': './dataset/SMCI_10cm_2010-2018',
            'CoLM': './dataset/CoLM_SoilMoisture_2010_2018.npy'
        }
        self.station_data_path = './dataset/soil_moisture_corrected_continuous_range.npy'

        self.data_start_date = '2010-01-01'
        self.data_end_date = '2018-12-31'
 
        self.train_start_date = '2010-01-01'
        self.train_end_date = '2016-12-31'
        self.val_start_date = '2017-01-01'
        self.val_end_date = '2017-12-31'
        self.test_start_date = '2018-01-01'
        self.test_end_date = '2018-12-31'



class ModelConfig:
    def __init__(self):
        self.input_dim = 12
        self.shared_dim = 128
        # self.dynamic_channels = 7 
        # self.static_channels = 5
        # self.lstm_hidden_dim = 512
        # self.lstm_num_layers = 2
        # self.dropout = 0.2
        # self.fusion_hidden_dim =128
        # self.forecast_horizon = 7
        # self.target_resolution = (160, 200)






PRODUCT_CONFIGS = {
    'era5': {
        'resolution': (160, 200),
        'filepath':'./dataset/ERA5_Land_SoilMoisture_2010-2018.npy',
        'has_missing': False,
        'weight': 1.0
    },
    'smci': {
        'resolution': (160,200),
        'filepath':'./dataset/SMCI_10cm_2010-2018.npy',
        'has_missing': False,
        'weight': 1.0
    },
    'colm': {
        'resolution': (160, 200),
        'filepath':'./dataset/CoLM_SoilMoisture_2010_2018.npy',
        'has_missing': False,
        'weight': 1.0
    }
}
