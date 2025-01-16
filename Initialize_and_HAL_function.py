import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import math

def initialize_GS(init_num, scaled_X_train, y_train, mode='Euclidean'):
    centroid = np.mean(scaled_X_train, axis=0)
    Z_init_x = []
    Z_init_y = []
    init_idx_GS = []

    dist_df = pd.DataFrame([math.dist(i, centroid) for i in scaled_X_train], columns=['Dist'])
    idx = dist_df.sort_values(by='Dist', ascending=True).index.tolist()[0]
    Z_init_x.append(scaled_X_train[idx])
    Z_init_y.append(np.array([y_train[idx]]))
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
            # if mode == 'Cosine':
            #     d_nm = [1 / cos_sim(sample, scaled_X_train_new_not_selected.iloc[i]) for sample in Z_init_x]
            d_n.append(np.min(d_nm))
        d_n_df = pd.DataFrame(d_n, columns=['Dist'])
        idx = d_n_df.sort_values(by='Dist', ascending=False).index.tolist()[0]
        init_idx_GS.append(original_idx[idx])
        # print('Original idx : {}, idx : {}'.format(original_idx[idx], idx))
        del original_idx[idx]

        Z_init_x.append(scaled_X_train_new_not_selected.iloc[idx].to_numpy())
        Z_init_y.append(y_train_new_not_selected.iloc[idx].to_numpy())

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
        sampling_num, round_num):
    pred = []
    test_mae = []
    test_mse = []
    test_r2 = []

    d_selected = xgb.DMatrix(X_selected, y_selected)
    d_not_selected = xgb.DMatrix(X_not_selected, y_not_selected)
    d_test = xgb.DMatrix(scaled_X_test, y_test)
    for param in p_list:
        param['device'] = 'cuda:1'
        param['nthread'] = 4
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

    M_param = ((day) / round_num) / 2
    M_I_param = 1 - M_param
    print(round(M_I_param, 3), round(M_param, 3))

    u_score = np.stack(pred).std(axis=0)
    normalized_u_score = u_score / max(u_score)

    for k in range(sampling_num):
        d_n = []
        for i in range(X_not_selected.shape[0]):
            #if distance == 'Euclidean':
            d_nm = [math.dist(sample, X_not_selected[i]) for sample in X_selected]
            #if distance == 'Cosine':
            #    d_nm = [1 / cos_sim(sample, X_not_selected[i]) for sample in X_selected]
            d_n.append(np.min(d_nm))
        normalized_d_n = d_n / (np.max(d_n) + 1e-8)
        d_n_df = pd.DataFrame(M_param * normalized_u_score + M_I_param * normalized_d_n, columns=['HAL_score'])
        idx = d_n_df.sort_values(by='HAL_score', ascending=False).index.tolist()[0]

        X_selected = np.concatenate((X_selected, X_not_selected[idx].reshape(1, -1)), axis=0)
        y_selected = np.concatenate((y_selected.reshape(-1, 1), y_not_selected[idx].reshape(-1, 1)), axis=0)
        normalized_u_score = np.delete(normalized_u_score, idx, axis=0)

        X_not_selected = np.delete(X_not_selected, idx, axis=0)
        y_not_selected = np.delete(y_not_selected, idx, axis=0)

    return X_selected, y_selected, X_not_selected, y_not_selected, (test_mae, test_mse, test_r2)
