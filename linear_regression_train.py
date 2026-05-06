"""
train_linear_models.py


"""

import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

def load_train_features(base_dir='./eval_data_1'):

    train_feat1_era5 = np.load(os.path.join(base_dir, 'train_feat_stage1_era5.npy'))
    train_feat1_colm = np.load(os.path.join(base_dir, 'train_feat_stage1_colm.npy'))
    train_feat1_smci = np.load(os.path.join(base_dir, 'train_feat_stage1_smci.npy'))
    train_feat2_cons = np.load(os.path.join(base_dir, 'train_feat_stage2_consensus.npy'))
    train_obs_era5 = np.load(os.path.join(base_dir, 'train_obs_era5_day1.npy'))
    train_obs_colm = np.load(os.path.join(base_dir, 'train_obs_colm_day1.npy'))
    train_obs_smci = np.load(os.path.join(base_dir, 'train_obs_smci_day1.npy'))

    return (train_feat1_era5, train_feat1_colm, train_feat1_smci, train_feat2_cons,
            train_obs_era5, train_obs_colm, train_obs_smci)

def train_and_save():
    os.makedirs('./linear_models', exist_ok=True)


    (tr_f1_era5, tr_f1_colm, tr_f1_smci, tr_f2_cons,
     tr_obs_era5, tr_obs_colm, tr_obs_smci) = load_train_features()


    targets_info = {
        'ERA5': (tr_obs_era5,
                 {'Stage1_ERA5': tr_f1_era5,
                  'Stage1_CoLM': tr_f1_colm,
                  'Stage1_SMCI': tr_f1_smci,
                  'Stage2_Consensus': tr_f2_cons}),
        'CoLM': (tr_obs_colm,
                 {'Stage1_ERA5': tr_f1_era5,
                  'Stage1_CoLM': tr_f1_colm,
                  'Stage1_SMCI': tr_f1_smci,
                  'Stage2_Consensus': tr_f2_cons}),
        'SMCI': (tr_obs_smci,
                 {'Stage1_ERA5': tr_f1_era5,
                  'Stage1_CoLM': tr_f1_colm,
                  'Stage1_SMCI': tr_f1_smci,
                  'Stage2_Consensus': tr_f2_cons}),
    }

    for target_name, (y_tr, feature_dict) in targets_info.items():

        label_mask = ~np.isnan(y_tr)


        for feat_name, X_tr in feature_dict.items():

            X_initial = X_tr[label_mask]
            y_initial = y_tr[label_mask]


            feature_row_mask = ~np.isnan(X_initial).any(axis=1)
            X_clean = X_initial[feature_row_mask]
            y_clean = y_initial[feature_row_mask]


            if X_clean.shape[0] == 0:

                continue


            model = Ridge(alpha=1.0)
            model.fit(X_clean, y_clean)


            y_pred = model.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))



            filename = f"./linear_models/ridge_{target_name}_{feat_name}.pkl"
            joblib.dump(model, filename)


if __name__ == '__main__':
    train_and_save()
