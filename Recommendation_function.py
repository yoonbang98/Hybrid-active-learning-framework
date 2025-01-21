import pandas as pd
import numpy as np
import xgboost as xgb

from tqdm import tqdm
import math

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from itertools import product

onehot_col_list = ['SlIdi', 'AtIdi', 'RcIdi', 'OgIdi',
                   'EcispA', 'KaispA', 'JcispA', 'SeispA', 'BaispA',
                   'PacrtE', 'KpcrtE', 'HpcrtE', 'NicrtE', 'ShcrtE',
                   'PacrtB', 'KpcrtB', 'LacrtB', 'RccrtB', 'SicrtB',
                   'PacrtI', 'ErcrtI', 'SscrtI', 'BvcrtI', 'LacrtI']
homolog_num_dict = {"none" : 0, "SlIdi":1, 'AtIdi':2, 'RcIdi':3, 'OgIdi':4,
                   'EcispA':1, 'KaispA':2, 'JcispA':3, 'SeispA':4, 'BaispA':5,
                   'PacrtE':1, 'KpcrtE':2, 'HpcrtE':3, 'NicrtE':4, 'ShcrtE':5,
                   'PacrtB':1, 'KpcrtB':2, 'LacrtB':3, 'RccrtB':4, 'SicrtB':5,
                   'PacrtI':1, 'ErcrtI':2, 'SscrtI':3, 'BvcrtI':4, 'LacrtI':5}
exploration_param = {1: 1.41, 2: 1.41, 3: 1.41, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.5,
                     9: 0.5, 10: 0.5 , 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5,
                     17: 0.5, 18: 0.5, 19: 0.5, 20: 0.5}
ensemble_len = 20
# Create the grid search parameter and scoring functions
param_grid = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "max_depth": [2, 3, 4, 6 ,8],
    "objective": ['reg:absoluteerror'],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.4, 0.6],
    "min_child_weight": [1, 2, 4],
    "random_state" : [42],
    "nthread" : [2]}

def preprocess(df, label, col_list): #Preprocessing
    df_processed = []
    label_processed = []
    for group in df.groupby(df.columns.tolist()):
        idx_list = group[-1].index.tolist()  # group index info
        iter_num = len(idx_list) // 3
        for i in range(iter_num + 1):
            try:
                tmp = idx_list[i * 3:(i + 1) * 3]
            except:
                tmp = idx_list[i * 3:]
            if len(tmp) > 0:
                df_processed.append(group[-1].iloc[0, :])
                label_processed.append(np.mean(label[tmp]))
    df_processed = pd.DataFrame(np.stack(df_processed), columns=col_list)
    label_processed = pd.Series(label_processed)
    return df_processed, label_processed

def make_onehot(df): #One-hot encoding
    enz_c = df[['Idi', 'ispA', 'CrtE', 'CrtB', 'CrtI']]
    enz_amount = df[['Idi_conc', 'ispA_conc', 'CrtE_conc', 'CrtB_conc', 'CrtI_conc']]

    enz_onehot = []
    for _, row in enz_c.iterrows():
        tmp = []
        for idx, enz in enumerate(['Idi', 'ispA', 'CrtE', 'CrtB', 'CrtI']):
            if enz == 'Idi':
                onehot = [0] * 4
                if int(row[enz]) < 1:
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

    df_onehot = pd.concat([enz_onehot, enz_amount], axis=1)
    return df_onehot

def generate_all_possible_conditions(): #Generate all possible lycopene experiment conditions
    col_list = onehot_col_list + ['Idi_conc', 'ispA_conc', 'CrtE_conc', 'CrtB_conc', 'CrtI_conc']
    ispa_conc_list = [0, 0.1, 0.25, 0.5, 1]
    conc_list = [0.1, 0.25, 0.5, 1]

    idi_enz_type = product([1, 2, 3, 4], repeat=1)
    enz_type = product([1, 2, 3, 4, 5], repeat=4)

    ispa_enz_conc = product(ispa_conc_list, repeat=1)
    enz_conc = product(conc_list, repeat=4)

    num = 0
    ALL_possible_conditions = []
    for i in tqdm(product(idi_enz_type, ispa_enz_conc, enz_type, enz_conc)):
        one_hot_encoded = np.zeros((1, 24))

        enz_type_tmp1 = np.array(i[0])
        enz_conc_tmp1 = np.array(i[1])
        enz_type_tmp2 = np.array(i[2])
        enz_conc_tmp2 = np.array(i[3])

        if enz_conc_tmp1[0] == 0:
            enz_type_tmp2[0] = 0
        one_hot_encoded[0, enz_type_tmp1[0] - 1] = 1
        for j, value in enumerate(enz_type_tmp2):
            if value > 0:
                one_hot_encoded[0, 4 + 5 * j + value - 1] = 1

        concat_tmp = np.concatenate((one_hot_encoded, enz_conc_tmp2[:1].reshape(1, -1), enz_conc_tmp1.reshape(1, -1),
                                     enz_conc_tmp2[1:].reshape(1, -1)), axis=1)

        ALL_possible_conditions.append(concat_tmp)
        num += 1

    ALL_possible_conditions = pd.DataFrame(np.squeeze(np.array(ALL_possible_conditions)), columns=col_list)
    ALL_possible_conditions.drop_duplicates(inplace=True)
    ALL_possible_conditions = ALL_possible_conditions.reset_index(drop=True)
    print('Number of all possible experimental conditions : {}'.format(ALL_possible_conditions.shape[0]))
    return ALL_possible_conditions

def optimize_param(x, y): # Hyperparameter tuning
    model = XGBRegressor(n_estimators=500)

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        n_iter=1000,
        random_state=42)
    grid.fit(x, y)
    grid_results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
    print(-1 * np.mean(grid_results['mean_test_score'][:ensemble_len]))
    params_list = grid_results.params.iloc[0:ensemble_len, ].tolist()
    return params_list, grid_results['mean_test_score'][:]

def transform_df(df):
    df_conc = df.iloc[:,-5:]
    df_type = []
    for idx, row in df.iterrows():
        row_type = row[:-5]
        type_tmp = row_type[row_type!=0].index.tolist()
        if len(type_tmp) != 5:
            type_tmp = type_tmp[:1] + ['none'] + type_tmp[1:]
        type_tmp_num = []
        for t in type_tmp:
            type_tmp_num.append(homolog_num_dict[t])
        df_type.append(type_tmp_num)
    df_type = pd.DataFrame(df_type, columns =['Idi', 'ispA', 'CrtE','CrtB','CrtI'])
    df_transformed = pd.concat([df_type, df_conc], axis = 1)
    return df_transformed

def test_func(p_list, x_train, x_test, y_train, y_test):
    test_mae = []
    test_pred_total = []
    d_train_onehot = xgb.DMatrix(x_train, y_train)
    d_test_onehot = xgb.DMatrix(x_test, y_test)
    for params in p_list:
        params['device'] = 'cuda:1'
        model = xgb.train(params, d_train_onehot, num_boost_round=500)

        test_pred = model.predict(d_test_onehot)
        test_pred_total.append(test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        test_mae.append(mae)
    print('Test score : {}'.format(np.mean(test_mae)))
    return test_mae, test_pred_total

def METIS_recommendation(p_list, X_selected, y_selected, X_not_selected, sampling_num):
    pred = []

    d_selected = xgb.DMatrix(X_selected, y_selected)
    d_not_selected = xgb.DMatrix(X_not_selected, np.zeros(X_not_selected.shape[0]))
    for param in p_list:
        param['device'] = 'cuda:1'
        param['nthread'] = 2

        model = xgb.train(param, d_selected, num_boost_round=500)
        temp = model.predict(d_not_selected)
        pred.append(temp)

    exploration, exploitation = 1, 0
    ucb_score = exploration * np.stack(pred).std(axis=0) + exploitation * np.stack(pred).mean(axis=0)
    ucb_df = pd.DataFrame(ucb_score, columns=['UCB'])

    next_round_idx = ucb_df.sort_values(by='UCB', ascending=False).index.tolist()[:sampling_num]
    idx_list = []
    for idx in ucb_df.sort_values(by='UCB', ascending=False).index.tolist():
        target_row = X_not_selected[idx]
        row_exists = np.any(np.all(X_selected == target_row, axis=1))
        if row_exists:
            continue
        else:
            idx_list.append(idx)
        if len(idx_list) == sampling_num:
            break

    return X_not_selected[next_round_idx]

def HAL_recommendation(p_list, X_selected, y_selected, X_not_selected, day, sampling_num, round_num):
    pred = []
    d_selected = xgb.DMatrix(X_selected, y_selected)
    d_not_selected = xgb.DMatrix(X_not_selected, np.zeros(X_not_selected.shape[0]))

    for param in p_list:
        param['device'] = 'cuda:1'
        param['nthread'] = 2

        model = xgb.train(param, d_selected, num_boost_round=500)
        temp = model.predict(d_not_selected)
        pred.append(temp)

    M_param = ((day) / round_num) / 2
    M_I_param = 1 - M_param
    print(round(M_I_param, 3), round(M_param, 3))

    u_score = np.stack(pred).std(axis=0)
    normalized_u_score = u_score / max(u_score)

    for k in range(sampling_num):
        d_n = []
        for i in tqdm(range(X_not_selected.shape[0])):
            d_nm = [math.dist(sample, X_not_selected[i]) for sample in X_selected]
            d_n.append(np.min(d_nm))
        normalized_d_n = d_n / np.max(d_n)
        d_n_df = pd.DataFrame(M_param * normalized_u_score + M_I_param * normalized_d_n,
                              columns=['HAL_score'])  # HAL score calculation
        idx = d_n_df.sort_values(by='HAL_score', ascending=False).index.tolist()[0]
        print(X_not_selected[idx].reshape(1, -1))
        X_selected = np.concatenate((X_selected, X_not_selected[idx].reshape(1, -1)), axis=0)
        X_not_selected = np.delete(X_not_selected, idx, axis=0)
        normalized_u_score = np.delete(normalized_u_score, idx, axis=0)

    return X_selected[-sampling_num:]
