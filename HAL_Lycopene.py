#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def make_data(seed_list, enz_df, result_df): # Preprocessing
    train_x_onehot_total = []
    test_x_onehot_total = []

    train_y_total = []
    test_y_total = []

    for s in seed_list:
        np.random.seed(s)
        test_idx = np.random.choice([i for i in range(result_df.shape[0])], round(result_df.shape[0] * 0.2),
                                    replace=False)  # 20% of data for test
        test_label = result_df.iloc[test_idx]
        test_enz_onehot = enz_df.iloc[test_idx, :]

        train_label = result_df.drop(test_idx).reset_index(drop=True)
        train_enz_onehot = enz_df.drop(test_idx).reset_index(drop=True)

        scaler2 = MinMaxScaler()
        scaled_X_train_onehot = scaler2.fit_transform(train_enz_onehot)
        scaled_X_test_onehot = scaler2.transform(test_enz_onehot)

        train_x_onehot_total.append(scaled_X_train_onehot)
        test_x_onehot_total.append(scaled_X_test_onehot)

        train_y_total.append(train_label)
        test_y_total.append(test_label)

    print('X_train_onehot shape : {}'.format(scaled_X_train_onehot.shape))
    print('X_test_onehot shape : {}'.format(scaled_X_test_onehot.shape))
    print('y_train : {}, y_test : {}'.format(len(train_label), len(test_label)))
    return train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total

if __name__ == "__main__":
    from Initialize_and_HAL_function import initialize_GS, HAL, optimize_param

    lycopene_data = pd.read_csv('/work/home/ybchae/active_learning/data/lycopene/IPP_only_normalized_yb.csv',
                                index_col=0)
    label = lycopene_data['result']
    lycopene_data.drop(columns=['result'], inplace=True)
    col_list = lycopene_data.columns.tolist()

    lycopene_processed = []
    label_processed = []
    for group in lycopene_data.groupby(lycopene_data.columns.tolist()):
        idx_list = group[-1].index.tolist()  # group index
        iter_num = len(idx_list) // 3  # triplet preprocessing
        for i in range(iter_num + 1):
            try:
                tmp = idx_list[i * 3:(i + 1) * 3]
            except:
                tmp = idx_list[i * 3:]
            if len(tmp) > 0:
                lycopene_processed.append(group[-1].iloc[0, :])
                label_processed.append(np.mean(label[tmp]))
    lycopene_processed = pd.DataFrame(np.stack(lycopene_processed), columns=col_list)
    label_processed = pd.Series(label_processed)

    enz_c = lycopene_processed[['Idi', 'ispA', 'CrtE', 'CrtB', 'CrtI']]
    enz_amount = lycopene_processed[['Idi_conc', 'ispA_conc', 'CrtE_conc', 'CrtB_conc', 'CrtI_conc']]

    enz_onehot = []
    onehot_col_list = ['SlIdi', 'AtIdi', 'RcIdi', 'OgIdi',
                       'EcispA', 'KaispA', 'JcispA', 'SeispA', 'BaispA',
                       'PacrtE', 'KpcrtE', 'HpcrtE', 'NicrtE', 'ShcrtE',
                       'PacrtB', 'KpcrtB', 'LacrtB', 'RccrtB', 'SicrtB',
                       'PacrtI', 'ErcrtI', 'SscrtI', 'BvcrtI', 'LacrtI']

    for _, row in enz_c.iterrows(): #One hot encoding
        tmp = []
        for idx, enz in enumerate(['Idi', 'ispA', 'CrtE', 'CrtB', 'CrtI']):
            if enz == 'Idi':
                onehot = [0] * 4
                if int(row[enz]) == 0:
                    tmp.extend(onehot)
                else:
                    onehot[int(row[enz]) - 1] = 1
                    tmp.extend(onehot)
            else:
                onehot = [0] * 5
                if int(row[enz]) == 0:
                    tmp.extend(onehot)
                else:
                    onehot[int(row[enz]) - 1] = 1
                    tmp.extend(onehot)
        enz_onehot.append(tmp)
    enz_onehot = pd.DataFrame(enz_onehot, columns=onehot_col_list)
    lycopene_processed_onehot = pd.concat([enz_onehot, enz_amount], axis=1)

    train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total = make_data([12,22,32,42,52,62,72,82,92,102], lycopene_processed_onehot, label_processed)

    num_iter = 1000

    init_num = 20
    sampling_num = 10
    HAL_v2_total = []

    random_seed = [12,22,32,42,52,62,72,82,92,102]
    for i in tqdm(range(len(random_seed))):
        seed = random_seed[i]
        np.random.seed(seed)

        scaled_X_train = train_x_onehot_total[i]
        y_train = train_y_total[i]
        scaled_X_test = test_x_onehot_total[i]
        y_test = test_y_total[i]

        X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS = initialize_GS(
            init_num, scaled_X_train, y_train, 'Euclidean')
        round_num = 6

        HAL_v2_seed = []
        param_list_GS = optimize_param(X_train_init_GS, y_train_init_GS, seed, num_iter)

        for d in range(round_num):
            if d == 0:
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(
                    param_list_GS, 1, X_train_init_GS, y_train_init_GS,
                    scaled_X_train_new_not_selected, y_train_new_not_selected,
                    scaled_X_test, y_test, sampling_num, round_num)
                HAL_v2_seed.append(HAL_v2_mae)
                print('Day {} HAL v2 MAE value : {}'.format(d + 1, np.mean(HAL_v2_mae[0])))
                print('--------------------------------------------')

            else:
                param_list_HAL_v2 = optimize_param(X_new_HAL_v2, y_new_HAL_v2, seed, num_iter)
                print('HAL2 optim done!')
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(
                    param_list_HAL_v2, d + 1, X_new_HAL_v2, y_new_HAL_v2,
                    X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2,
                    scaled_X_test, y_test, sampling_num, round_num)
                if len(HAL_v2_mae) > 0:
                    HAL_v2_seed.append(HAL_v2_mae)
                    print('Day {} HAL v2 MAE value : {}'.format(d + 1, np.mean(HAL_v2_mae[0])))
                    print('--------------------------------------------')
                    print(X_new_HAL_v2.shape)
        HAL_v2_total.append(HAL_v2_seed)