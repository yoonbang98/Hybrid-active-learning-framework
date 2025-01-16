#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from tqdm import tqdm

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

if __name__ == "__main__":
    from Initialize_and_HAL_function import initialize_GS, HAL, optimize_param
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

    num_iter = 1000

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

        X_train_init_GS, y_train_init_GS, scaled_X_train_new_not_selected, y_train_new_not_selected, init_idx_GS = initialize_GS(init_num, scaled_X_train, y_train, 'Euclidean')
        round_num = 6

        HAL_v2_seed = []
        param_list_GS = optimize_param(X_train_init_GS, y_train_init_GS, seed, num_iter)

        for d in range(round_num):
            if d == 0 :
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(param_list_GS, 1,X_train_init_GS, y_train_init_GS,
                                                                                                                   scaled_X_train_new_not_selected, y_train_new_not_selected,
                                                                                                                   scaled_X_test, y_test, sampling_num, round_num)
                HAL_v2_seed.append(HAL_v2_mae)
                print('Day {} HAL v2 MAE value : {}'.format(d+1,np.mean(HAL_v2_mae[0])))
                print('--------------------------------------------')

            else :
                param_list_HAL_v2 = optimize_param(X_new_HAL_v2, y_new_HAL_v2, seed, num_iter)
                print('HAL2 optim done!')
                X_new_HAL_v2, y_new_HAL_v2, X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2, HAL_v2_mae = HAL(param_list_HAL_v2, d+1,X_new_HAL_v2, y_new_HAL_v2,
                                                                                                                   X_not_selected_new_HAL_v2, y_not_selected_new_HAL_v2,
                                                                                                                   scaled_X_test, y_test, sampling_num, round_num)
                if len(HAL_v2_mae) > 0 :
                    HAL_v2_seed.append(HAL_v2_mae)
                    print('Day {} HAL v2 MAE value : {}'.format(d+1,np.mean(HAL_v2_mae[0])))
                    print('--------------------------------------------')
                    print(X_new_HAL_v2.shape)
        HAL_v2_total.append(HAL_v2_seed)
