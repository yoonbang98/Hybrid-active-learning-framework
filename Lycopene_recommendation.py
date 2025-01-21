if __name__ == "__main__":
    import os
    import pandas as pd

    from sklearn.preprocessing import MinMaxScaler
    from Recommendation_function import preprocess, make_onehot, optimize_param, transform_df, \
        generate_all_possible_conditions, METIS_recommendation, HAL_recommendation

    path = os.getcwd()
    lycopene_data = pd.read_csv(path + '/data/lycopene/IPP_only_normalized_yb.csv',
                                index_col=0)
    label = lycopene_data['result']
    lycopene_data.drop(columns=['result'], inplace=True)
    col_list = lycopene_data.columns.tolist()

    lycopene_processed, label_processed = preprocess(lycopene_data, label, col_list)
    lycopene_processed_onehot = make_onehot(lycopene_processed)

    onehot_col_list = ['SlIdi', 'AtIdi', 'RcIdi', 'OgIdi',
                       'EcispA', 'KaispA', 'JcispA', 'SeispA', 'BaispA',
                       'PacrtE', 'KpcrtE', 'HpcrtE', 'NicrtE', 'ShcrtE',
                       'PacrtB', 'KpcrtB', 'LacrtB', 'RccrtB', 'SicrtB',
                       'PacrtI', 'ErcrtI', 'SscrtI', 'BvcrtI', 'LacrtI']

    random_test = pd.read_csv(path + '/data/lycopene/random_test_set_result.csv')
    random_label = random_test['result'].apply(lambda x: max(0, x))
    random_test.drop(columns=['result'], inplace=True)
    random_test_processed, random_label_processed = preprocess(random_test, random_label, col_list)
    random_test_processed_onehot = make_onehot(random_test_processed)

    ALL_possible_conditions = generate_all_possible_conditions()

    scaler = MinMaxScaler()
    scaled_ALL_possible_conditions = scaler.fit_transform(ALL_possible_conditions)
    scaled_X_train = scaler.transform(lycopene_processed_onehot)
    scaled_X_test = scaler.transform(random_test_processed_onehot)

    params_list, grid_results = optimize_param(scaled_X_train, label_processed)

    METIS_next = METIS_recommendation(params_list, scaled_X_train, label_processed, scaled_ALL_possible_conditions, sampling_num=10)
    HAL_next = HAL_recommendation(params_list, scaled_X_train, label_processed, scaled_ALL_possible_conditions, day=1, sampling_num=10, round_num=5)

    col_list2 = onehot_col_list + ['Idi_conc','ispA_conc','CrtE_conc','CrtB_conc','CrtI_conc']
    METIS_next_df = pd.DataFrame(scaler.inverse_transform(METIS_next), columns=col_list2)
    HAL_next_df = pd.DataFrame(scaler.inverse_transform(HAL_next), columns=col_list2)

    METIS_next_df_transformed = transform_df(METIS_next_df)
    HAL_next_df_transformed = transform_df(HAL_next_df)

    METIS_next_df_transformed[
        ['Idi', 'Idi_conc', 'ispA', 'ispA_conc', 'CrtE', 'CrtE_conc', 'CrtB', 'CrtB_conc', 'CrtI', 'CrtI_conc']].to_csv(
        path + '/data/METIS_Day1.csv', index=False)
    HAL_next_df_transformed[
        ['Idi', 'Idi_conc', 'ispA', 'ispA_conc', 'CrtE', 'CrtE_conc', 'CrtB', 'CrtB_conc', 'CrtI', 'CrtI_conc']].to_csv(
        path + '/data/HAL_Day1.csv', index=False)