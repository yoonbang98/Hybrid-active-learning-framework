import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import math

ensemble_len = 20
# Create the grid search parameter and scoring functions
param_grid = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "max_depth": [2, 3, 4, 6, 8],
    "objective": ['reg:absoluteerror'],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.4, 0.6],
    "min_child_weight": [1, 2, 4],
    "random_state": [42],
    "nthread": [2]}

def optimize_param(x, y, seed, num_iter): # Hyperparameter tuning
    model = XGBRegressor(n_estimators=100)

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        n_iter=num_iter,
        random_state=seed)
    grid.fit(x, y)
    grid_results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
    params_list = grid_results.params.iloc[0:ensemble_len, ].tolist()
    return params_list

def initialize_GS(init_num, scaled_X_train, y_train, mode='Euclidean'): # Initialize using Greedy sampling
    centroid = np.mean(scaled_X_train, axis=0)
    Z_init_x = []
    Z_init_y = []
    init_idx_GS = []

    dist_df = pd.DataFrame([math.dist(i, centroid) for i in scaled_X_train], columns=['Dist'])
    idx = dist_df.sort_values(by='Dist', ascending=True).index.tolist()[0]
    Z_init_x.append(scaled_X_train[idx])

    Z_init_y.append(np.array([y_train[idx]]).reshape(-1,))

    init_idx_GS.append(idx)

    scaled_X_train_new_not_selected = pd.DataFrame(scaled_X_train).drop([idx], axis=0)
    y_train_new_not_selected = pd.DataFrame(y_train).drop([idx], axis=0)

    original_idx = scaled_X_train_new_not_selected.index.tolist()

    scaled_X_train_new_not_selected.reset_index(inplace=True, drop=True)
    y_train_new_not_selected.reset_index(inplace=True, drop=True)

    for k in range(init_num - 1):
        d_n = []
        for i in range(scaled_X_train_new_not_selected.shape[0]):
            if mode == 'Euclidean':
                d_nm = [math.dist(sample, scaled_X_train_new_not_selected.iloc[i]) for sample in Z_init_x]
            d_n.append(np.min(d_nm))
        d_n_df = pd.DataFrame(d_n, columns=['Dist'])
        idx = d_n_df.sort_values(by='Dist', ascending=False).index.tolist()[0]
        init_idx_GS.append(original_idx[idx])
        del original_idx[idx]

        Z_init_x.append(scaled_X_train_new_not_selected.iloc[idx].to_numpy())
        Z_init_y.append(y_train_new_not_selected.iloc[idx].to_numpy().reshape(-1,))

        scaled_X_train_new_not_selected = scaled_X_train_new_not_selected.drop(idx, axis=0)
        y_train_new_not_selected = y_train_new_not_selected.drop(idx, axis=0)

        scaled_X_train_new_not_selected.reset_index(inplace=True, drop=True)
        y_train_new_not_selected.reset_index(inplace=True, drop=True)

    X_train_init_GS = np.stack(Z_init_x, axis=0)
    y_train_init_GS = np.stack(Z_init_y, axis=0)
    scaled_X_train_new_not_selected = scaled_X_train_new_not_selected.to_numpy()
    y_train_new_not_selected = y_train_new_not_selected.to_numpy()

    return X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS


def HAL(p_list, day, X_selected, y_selected, X_not_selected, y_not_selected, scaled_X_test, y_test,
        sampling_num, round_num): # HAL function
    pred = []
    test_mae = []
    test_mse = []
    test_r2 = []

    d_selected = xgb.DMatrix(X_selected, y_selected)
    d_not_selected = xgb.DMatrix(X_not_selected, y_not_selected)
    d_test = xgb.DMatrix(scaled_X_test, y_test)
    for param in p_list:
        param['device'] = 'cuda:1'
        #param['nthread'] = 4
        model = xgb.train(param, d_selected, num_boost_round=100)

        temp = model.predict(d_not_selected)
        temp2 = model.predict(d_test)

        pred.append(temp)
        mae_tmp = mean_absolute_error(y_test, temp2)
        mse_tmp = mean_squared_error(y_test, temp2)
        r2_tmp = r2_score(y_test, temp2)
        test_mae.append(mae_tmp)
        test_mse.append(mse_tmp)
        test_r2.append(r2_tmp)

    if X_not_selected.shape[0] == 0:
        return X_selected, y_selected, X_not_selected, y_not_selected, (test_mae, test_mse, test_r2)

    if sampling_num > X_not_selected.shape[0]:
        idx = [i for i in range(X_not_selected.shape[0])]
        X_selected = np.concatenate((X_selected, X_not_selected[idx]), axis=0)
        y_selected = np.concatenate((y_selected.reshape(-1, 1), y_not_selected[idx].reshape(-1, 1)), axis=0)

        X_not_selected = np.delete(X_not_selected, idx, axis=0)
        y_not_selected = np.delete(y_not_selected, idx, axis=0)
        return X_selected, y_selected, X_not_selected, y_not_selected, (test_mae, test_mse, test_r2)

    M_param = ((day) / round_num) / 2 # Uncertainty ratio
    M_I_param = 1 - M_param # Diversity ratio
    print(round(M_I_param, 3), round(M_param, 3))

    u_score = np.stack(pred).std(axis=0)
    normalized_u_score = u_score / max(u_score)

    for k in range(sampling_num):
        d_n = []
        for i in range(X_not_selected.shape[0]):
            d_nm = [math.dist(sample, X_not_selected[i]) for sample in X_selected]
            d_n.append(np.min(d_nm))
        normalized_d_n = d_n / (np.max(d_n) + 1e-8)
        d_n_df = pd.DataFrame(M_param * normalized_u_score + M_I_param * normalized_d_n, columns=['HAL_score']) # HAL score calculation
        idx = d_n_df.sort_values(by='HAL_score', ascending=False).index.tolist()[0]

        X_selected = np.concatenate((X_selected, X_not_selected[idx].reshape(1, -1)), axis=0)
        y_selected = np.concatenate((y_selected.reshape(-1, 1), y_not_selected[idx].reshape(-1, 1)), axis=0)
        normalized_u_score = np.delete(normalized_u_score, idx, axis=0)

        X_not_selected = np.delete(X_not_selected, idx, axis=0)
        y_not_selected = np.delete(y_not_selected, idx, axis=0)

    return X_selected, y_selected, X_not_selected, y_not_selected, (test_mae, test_mse, test_r2)
