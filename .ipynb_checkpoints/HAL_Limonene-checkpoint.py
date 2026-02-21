#!/usr/bin/env python
# coding: utf-8

def make_data(seed_list, enz_a, enz_c, result): # Preprocessing
    train_x_onehot_total = []
    test_x_onehot_total = []

    train_y_total = []
    test_y_total = []

    for s in seed_list:
        onehot_enc = OneHotEncoder(handle_unknown='ignore')

        a = onehot_enc.fit_transform(np.array(enz_c['1']).reshape(-1, 1))
        b = onehot_enc.fit_transform(np.array(enz_c['2']).reshape(-1, 1))
        c = onehot_enc.fit_transform(np.array(enz_c['3']).reshape(-1, 1))
        d = onehot_enc.fit_transform(np.array(enz_c['4']).reshape(-1, 1))
        e = onehot_enc.fit_transform(np.array(enz_c['5']).reshape(-1, 1))
        f = onehot_enc.fit_transform(np.array(enz_c['6']).reshape(-1, 1))
        g = onehot_enc.fit_transform(np.array(enz_c['7']).reshape(-1, 1))
        h = onehot_enc.fit_transform(np.array(enz_c['8']).reshape(-1, 1))
        i = onehot_enc.fit_transform(np.array(enz_c['9']).reshape(-1, 1))

        onehot_enz = np.concatenate((a.toarray(), b.toarray(), c.toarray(), d.toarray(), e.toarray(), f.toarray(),
                                     g.toarray(), h.toarray(), i.toarray()), axis=1)

        X_onehot = np.concatenate((onehot_enz, np.array(enz_a)), axis=1)
        X_train_onehot, X_test_onehot, y_train, y_test= train_test_split(X_onehot, np.array(result), test_size=0.2, random_state=s)

        scaler2 = MinMaxScaler()
        scaled_X_train_onehot = scaler2.fit_transform(X_train_onehot)
        scaled_X_test_onehot = scaler2.transform(X_test_onehot)


        train_x_onehot_total.append(scaled_X_train_onehot)
        test_x_onehot_total.append(scaled_X_test_onehot)

        train_y_total.append(y_train)
        test_y_total.append(y_test)

    print('X_train_onehot shape : {}'.format(scaled_X_train_onehot.shape))
    print('X_test_onehot shape : {}'.format(scaled_X_test_onehot.shape))
    print('y_train : {}, y_test : {}'.format(len(y_train), len(y_test)))
    return train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total


if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    from Initialize_and_HAL_function import initialize_GS, HAL, optimize_param

    path = os.getcwd()
    enz_amount = pd.read_csv(path + '/data/Limonene/limonene_enz_amount.csv')
    enz_comb = pd.read_csv(path + '/data/Limonene/limonene_enz_comb.csv')
    result = pd.read_csv(path + '/data/Limonene/limonene_titer.csv')

    enz_amount = enz_amount.fillna(0)
    drop_idx = set()
    for i in range(enz_comb.shape[1]):
        drop_idx = drop_idx | set(enz_comb.loc[enz_comb.iloc[:, i] == '#VALUE!'].index)
    drop_idx = list(drop_idx)

    enz_amount = enz_amount.drop(drop_idx).reset_index(drop=True)
    enz_comb = enz_comb.drop(drop_idx).reset_index(drop=True)
    result = result.drop(drop_idx).reset_index(drop=True)

    train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total = make_data([2,12,22,32,42, 52, 62, 72, 82, 92], enz_amount, enz_comb, result)

    num_iter = 200
    init_num = 30
    sampling_num = 30
    HAL_v2_total = []

    random_seed = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
    for i in tqdm(range(len(random_seed))):
        seed = random_seed[i]
        np.random.seed(seed)

        scaled_X_train = train_x_onehot_total[i]
        y_train = train_y_total[i]
        scaled_X_test = test_x_onehot_total[i]
        y_test = test_y_total[i]

        X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS = initialize_GS(
            init_num, scaled_X_train, y_train, 'Euclidean')
        round_num = 15

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
