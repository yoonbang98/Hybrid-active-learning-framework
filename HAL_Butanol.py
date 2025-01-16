#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from xgboost import XGBRegressor
from tqdm import tqdm
import math

def make_data(seed_list, enz_a, enz_c, result_df):
    train_x_onehot_total = []
    test_x_onehot_total = []
    
    train_y_total = []
    test_y_total = []
      
    for s in seed_list:
        np.random.seed(s)
        test_idx = np.random.choice([i for i in range(result_df.shape[0])], round(result_df.shape[0]*0.2), replace = False) # 20% of data for test
        
        test_label = result_df.iloc[test_idx, :]['24']
        test_enz_amount = enz_a.iloc[test_idx, :]
        test_enz_comb = enz_c.iloc[test_idx, :]

        train_label_tmp = result_df.drop(test_idx).reset_index(drop = True)
        train_enz_amount_tmp = enz_a.drop(test_idx).reset_index(drop = True)
        train_enz_comb_tmp = enz_c.drop(test_idx).reset_index(drop = True)
        
        train_2_idx = train_label_tmp['24_2'].loc[train_label_tmp['24_2'] != '-'].index.tolist() # 24_1, 24_2, 24_3을 하나로 합쳐 학습데이터를 만들려함  
        #단, 같은 효소 종류, 농도 데이터가 학습데이터와 검증 데이터에 나뉘면 안된다고 판단하여 먼저 24_1 기준으로 학습데이터와 검증 데이터를 나누고 학습 데이터 증강   
        train_3_idx = train_label_tmp['24_3'].loc[train_label_tmp['24_3'] != '-'].index.tolist()

        train_2_label = train_label_tmp.iloc[train_2_idx,:]['24_2']
        train_2_enz_amount = train_enz_amount_tmp.iloc[train_2_idx,:]
        train_2_enz_comb = train_enz_comb_tmp.iloc[train_2_idx,:]

        train_3_label = train_label_tmp.iloc[train_3_idx,:]['24_3']
        train_3_enz_amount = train_enz_amount_tmp.iloc[train_3_idx,:]
        train_3_enz_comb = train_enz_comb_tmp.iloc[train_3_idx,:]
        
        train_label = train_label_tmp['24']#pd.concat([train_label_tmp['24_1'],train_2_label,train_3_label],axis = 0) 
        train_enz_amount = train_enz_amount_tmp #pd.concat([train_enz_amount_tmp,train_2_enz_amount, train_3_enz_amount],axis = 0)
        train_enz_comb = train_enz_comb_tmp #pd.concat([train_enz_comb_tmp,train_2_enz_comb,train_3_enz_comb],axis = 0)
        
        onehot_enc = OneHotEncoder(handle_unknown='ignore')
        a = onehot_enc.fit_transform(np.array(train_enz_comb['1']).reshape(-1,1))
        a_test = onehot_enc.transform(np.array(test_enz_comb['1']).reshape(-1,1))

        b = onehot_enc.fit_transform(np.array(train_enz_comb['2']).reshape(-1,1))
        b_test = onehot_enc.transform(np.array(test_enz_comb['2']).reshape(-1,1))

        c = onehot_enc.fit_transform(np.array(train_enz_comb['3']).reshape(-1,1))
        c_test = onehot_enc.transform(np.array(test_enz_comb['3']).reshape(-1,1))

        d = onehot_enc.fit_transform(np.array(train_enz_comb['4']).reshape(-1,1))
        d_test = onehot_enc.transform(np.array(test_enz_comb['4']).reshape(-1,1))

        e = onehot_enc.fit_transform(np.array(train_enz_comb['5']).reshape(-1,1))
        e_test = onehot_enc.transform(np.array(test_enz_comb['5']).reshape(-1,1))

        train_onehot_enz = np.concatenate((a.toarray(),b.toarray(),c.toarray(),d.toarray(),e.toarray()), axis = 1)
        test_onehot_enz = np.concatenate((a_test.toarray(),b_test.toarray(),c_test.toarray(),d_test.toarray(),e_test.toarray()), axis = 1) 

        X_train_onehot = np.concatenate((train_onehot_enz, np.array(train_enz_amount)), axis = 1)
        X_test_onehot = np.concatenate((test_onehot_enz, np.array(test_enz_amount)), axis = 1)
        y_train_onehot = np.array(train_label, dtype = np.float64())
        y_test_onehot = np.array(test_label, dtype = np.float64())

        scaler2 = MinMaxScaler()
        scaled_X_train_onehot = scaler2.fit_transform(X_train_onehot)
        scaled_X_test_onehot = scaler2.transform(X_test_onehot)
        
        train_x_onehot_total.append(scaled_X_train_onehot)
        test_x_onehot_total.append(scaled_X_test_onehot)
        
        train_y_total.append(y_train_onehot)
        test_y_total.append(y_test_onehot)
        
    print('X_train_onehot shape : {}'.format(scaled_X_train_onehot.shape))
    print('X_test_onehot shape : {}'.format(scaled_X_test_onehot.shape))
    print('y_train : {}, y_test : {}'.format(len(y_train_onehot), len(y_test_onehot)))
    return train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total

# create the grid search object
def optimize_param(x,y, seed):
    model = XGBRegressor(n_estimators = 100)
    
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=5,
        scoring= 'neg_mean_absolute_error',
        n_jobs=-1,
        n_iter= 1000,
        random_state = seed)
    grid.fit(x, y)
    grid_results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
    #print(-1*np.mean(grid_results['mean_test_score'][:ensemble_len]))
    params_list = grid_results.params.iloc[0:ensemble_len,].tolist()
    return params_list

def initialize_GS(scaled_X_train, y_train, mode = 'Euclidean'):
    centroid = np.mean(scaled_X_train, axis = 0)
    Z_init_x = []
    Z_init_y = []
    init_idx_GS = []

    dist_df = pd.DataFrame([math.dist(i, centroid) for i in scaled_X_train], columns = ['Dist'])
    idx = dist_df.sort_values(by = 'Dist',ascending = True).index.tolist()[0]
    Z_init_x.append(scaled_X_train[idx])
    Z_init_y.append(np.array([y_train[idx]]))
    init_idx_GS.append(idx)

    scaled_X_train_new_not_selected = pd.DataFrame(scaled_X_train).drop([idx], axis = 0)
    y_train_new_not_selected = pd.DataFrame(y_train).drop([idx], axis = 0)

    original_idx = scaled_X_train_new_not_selected.index.tolist()

    scaled_X_train_new_not_selected.reset_index(inplace = True, drop = True)
    y_train_new_not_selected.reset_index(inplace = True, drop = True)

    for k in range(init_num-1):
        d_n = []
        for i in range(scaled_X_train_new_not_selected.shape[0]):
            if mode == 'Euclidean':
                d_nm = [math.dist(sample,scaled_X_train_new_not_selected.iloc[i]) for sample in Z_init_x]
            if mode == 'Cosine':
                d_nm = [1/cos_sim(sample,scaled_X_train_new_not_selected.iloc[i]) for sample in Z_init_x]
            d_n.append(np.min(d_nm))
        d_n_df = pd.DataFrame(d_n, columns = ['Dist'])
        idx = d_n_df.sort_values(by = 'Dist',ascending = False).index.tolist()[0]
        init_idx_GS.append(original_idx[idx])
        #print('Original idx : {}, idx : {}'.format(original_idx[idx], idx))
        del original_idx[idx]

        Z_init_x.append(scaled_X_train_new_not_selected.iloc[idx].to_numpy())
        Z_init_y.append(y_train_new_not_selected.iloc[idx].to_numpy())

        scaled_X_train_new_not_selected = scaled_X_train_new_not_selected.drop(idx, axis = 0)
        y_train_new_not_selected = y_train_new_not_selected.drop(idx, axis = 0)

        scaled_X_train_new_not_selected.reset_index(inplace = True, drop = True)
        y_train_new_not_selected.reset_index(inplace = True, drop = True)

    X_train_init_GS = np.stack(Z_init_x, axis = 0)
    y_train_init_GS = np.stack(Z_init_y, axis = 0)
    scaled_X_train_new_not_selected = scaled_X_train_new_not_selected.to_numpy()
    y_train_new_not_selected = y_train_new_not_selected.to_numpy()

    return X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS

def HAL(p_list, day, X_selected, y_selected, X_not_selected, y_not_selected, previous_r2, previous_mae,  mode = 'normal', distance = 'Euclidean'):
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

    if X_not_selected.shape[0] == 0 : 
        return X_selected, y_selected, X_not_selected, y_not_selected,(test_mae, test_mse, test_r2)

    if sampling_num > X_not_selected.shape[0]:
        idx = [i for i in range(X_not_selected.shape[0])]
        X_selected = np.concatenate((X_selected, X_not_selected[idx]), axis = 0)
        y_selected = np.concatenate((y_selected.reshape(-1,1), y_not_selected[idx].reshape(-1,1)), axis = 0)

        X_not_selected = np.delete(X_not_selected, idx, axis = 0)
        y_not_selected = np.delete(y_not_selected, idx, axis = 0)
        return X_selected, y_selected, X_not_selected, y_not_selected,(test_mae, test_mse, test_r2)
    
    if mode == 'normal':
        M_param = ((day)/round_num)/2
        M_I_param = 1 - M_param
    if mode == 'r2':
        if previous_r2 > 0:
            M_param = previous_r2
            M_I_param = 1 - M_param
        else :
            M_param, M_I_param = 0, 1
    if mode == 'loss':
        if previous_mae <= np.mean(test_mae):
            M_param, M_I_param = 0, 1
        else:
            M_param = ((day)/round_num)/2
            M_I_param = 1 - M_param
    print(round(M_I_param,3), round(M_param, 3))
    
    u_score = np.stack(pred).std(axis = 0)
    normalized_u_score = u_score/max(u_score)
    
    for k in range(sampling_num):
        d_n = []
        for i in range(X_not_selected.shape[0]):
            if distance == 'Euclidean':
                d_nm = [math.dist(sample,X_not_selected[i]) for sample in X_selected]
            if distance == 'Cosine':
                d_nm = [1/cos_sim(sample,X_not_selected[i]) for sample in X_selected]
            d_n.append(np.min(d_nm))
        normalized_d_n = d_n/(np.max(d_n)+1e-8)
        d_n_df = pd.DataFrame(M_param*normalized_u_score + M_I_param*normalized_d_n, columns = ['HAL_score'])
        idx = d_n_df.sort_values(by = 'HAL_score',ascending = False).index.tolist()[0]
        
        X_selected = np.concatenate((X_selected, X_not_selected[idx].reshape(1,-1)), axis = 0)
        y_selected = np.concatenate((y_selected.reshape(-1,1), y_not_selected[idx].reshape(-1,1)), axis = 0)
        normalized_u_score = np.delete(normalized_u_score, idx, axis = 0)

        X_not_selected = np.delete(X_not_selected, idx, axis = 0)
        y_not_selected = np.delete(y_not_selected, idx, axis = 0)

    return X_selected, y_selected, X_not_selected, y_not_selected,(test_mae, test_mse, test_r2)

if __name__ == "__main__":
    enz_amount = pd.read_csv('/work/home/ybchae/active_learning/data/iprobe/enzamount.csv', sep = '\t')
    enz_comb = pd.read_csv('/work/home/ybchae/active_learning/data/iprobe/enzcomb.csv',sep = '\t')
    result = pd.read_csv('/work/home/ybchae/active_learning/data/iprobe/exp_result.csv')

    lycopene_data = pd.read_csv('/work/home/ybchae/active_learning/data/lycopene/Lycopene_data.csv')

    lycopene_label = lycopene_data['a*(D65)']
    lycopene_label = lycopene_label.apply(lambda x: max(0, x))


    limonene_label = pd.read_csv('/work/home/ybchae/active_learning/processed_limonene_titer.csv')

    s1 = set(enz_amount.loc[enz_amount['1'] == 0].index.tolist())
    s2 = set(enz_amount.loc[enz_amount['2'] == 0].index.tolist())
    s3 = set(enz_amount.loc[enz_amount['3'] == 0].index.tolist())
    s4 = set(enz_amount.loc[enz_amount['4'] == 0].index.tolist())
    s5 = set(enz_amount.loc[enz_amount['5'] == 0].index.tolist())

    drop_idx = list(s1|s2|s3|s4|s5)

    enz_amount = enz_amount.drop(drop_idx).reset_index(drop = True)
    enz_comb = enz_comb.drop(drop_idx).reset_index(drop = True)
    result = result.drop(drop_idx).reset_index(drop = True)

    train_x_onehot_total, test_x_onehot_total, train_y_total, test_y_total = make_data([2,12,22,32,42,52,62,72,82,92], enz_amount, enz_comb, result)


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


    init_num = 20
    sampling_num = 20
    HAL_v2_total = []

    random_seed = [2,12,22,32,42,52,62,72,82,92]
    for i in tqdm(range(len(random_seed))) :
        seed = random_seed[i]
        np.random.seed(seed)

        scaled_X_train = train_x_onehot_total[i]
        y_train = train_y_total[i]
        scaled_X_test = test_x_onehot_total[i]
        y_test = test_y_total[i]

        X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS = initialize_GS(scaled_X_train, y_train, 'Euclidean')

        round_num = 6

        HAL_v2_seed = []
        param_list_GS = optimize_param(X_train_init_GS, y_train_init_GS, seed)

        for d in range(round_num):
            if d == 0 :
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(param_list_GS, 1,X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected,0, 2, 'normal', 'Euclidean')

                HAL_v2_seed.append(HAL_v2_mae)
                print('Day {} HAL v2 MAE value : {}'.format(d+1,np.mean(HAL_v2_mae[0])))
                print('--------------------------------------------')

            else :
                param_list_HAL_v2 = optimize_param(X_new_HAL_v2, y_new_HAL_v2, seed)
                print('HAL2 optim done!')
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(param_list_HAL_v2, d+1,X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2,np.mean(HAL_v2_mae[2]),np.mean(HAL_v2_mae[0]) ,'normal', 'Euclidean')

                if len(HAL_v2_mae) > 0 :
                    HAL_v2_seed.append(HAL_v2_mae)
                    print('Day {} HAL v2 MAE value : {}'.format(d+1,np.mean(HAL_v2_mae[0])))
                    print('--------------------------------------------')
                    print(X_new_HAL_v2.shape)
        HAL_v2_total.append(HAL_v2_seed)
