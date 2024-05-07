import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data_pneuma_test(PRMA, PRMP, scaler_x, scaler_y):
    df_list_MA_features = []
    df_list_MA_labels = []
    df_list_MP_features = []
    df_list_MP_features_2 = []
    df_list_MP_labels = []
    df_day_time = []

    edge_num = 3 ######
    num_features = 2
    num_cells = 6
    num_steps = 2

    days = ["20181024", "20181029", "20181030","20181101"]
    times = ["0800_0830", "0830_0900", "0900_0930", "0930_1000", "1000_1030", "1030_1100"]

    for day in days:
        for timetime in times:
            # some data is not good to use
            if not os.path.exists(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MA.csv"):
                continue
            if day == "20181101" and timetime == "0800_0830":
                continue
            if day == "20181029" and timetime == "1000_1030":
                continue
            if day == "20181030" and timetime == "0930_1000":
                continue
            if day == "20181030" and timetime == "1000_1030":
                continue
            # read data
            df_MA_features = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MA.csv")
            df_list_MA_features.append(df_MA_features)
            df_MP_features = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MP.csv")
            df_list_MP_features.append(df_MP_features)
            ### add feature 20231129
            df_MP_features_2 = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MP_add.csv")
            df_list_MP_features_2.append(df_MP_features_2)

            # df_MA_labels = pd.read_csv(f"../datasets/pneuma_corridor/backup/MP_{PRMP}_ma/{day}_{timetime}_MA_labels.csv")
            # df_list_MA_labels.append(df_MA_labels)
            df_MP_labels = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_label.csv")
            df_list_MP_labels.append(df_MP_labels)
            # save the length of data
            df_day_time.append(len(df_MA_features))

    df_X_MA = pd.concat(df_list_MA_features)
    # df_Y_MA = pd.concat(df_list_MA_labels)
    df_X_MP = pd.concat(df_list_MP_features)
    df_X_MP_2 = pd.concat(df_list_MP_features_2)
    df_Y_MP = pd.concat(df_list_MP_labels)
    # delete the first column
    df_X_MA = df_X_MA.drop(df_X_MA.columns[0], axis=1)
    # df_Y_MA = df_Y_MA.drop(df_Y_MA.columns[0], axis=1)
    df_X_MP = df_X_MP.drop(df_X_MP.columns[0], axis=1)
    df_X_MP_2 = df_X_MP_2.drop(df_X_MP_2.columns[0], axis=1)
    df_Y_MP = df_Y_MP.drop(df_Y_MP.columns[0], axis=1)

    # print(df_X_MA.head(), df_Y_MA.head(), df_X_MP.head(), df_Y_MP.head())
    # print(df_X_MA.shape, df_Y_MA.shape, df_X_MP.shape, df_Y_MP.shape)
    

    np_X_MA = df_X_MA.values.reshape(-1, 17, 2) # counts
    np_X_MA = np_X_MA[:, :, 1:2] # delete the input traffic counts

    np_X_MP = df_X_MP.values.reshape(-1, 17, 3) # total travel time and total travel distance
    # print(np_X_MP.shape, np_X_MP_2.shape, 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # np_X_MP_2 = df_X_MP_2.values.reshape(-1, 17, 32) # speed
    np_X_MP_2 = df_X_MP_2.values.reshape(-1, num_steps, 17, num_features*num_cells).transpose(0, 2, 1, 3).reshape(-1, 17, num_features*num_cells*num_steps) # speed
    
    np_X_MP = np_X_MP[:, :, [0,1]] # delete the speed

    np_Y_MP_qk = df_Y_MP.values.reshape(-1, 17, 3)
    np_Y_MP = np_Y_MP_qk[:, :, 1].reshape(-1, 17, 1)  # density veh/km
    # np_Y_MP = np_Y_MP_qk[:, :, 2].reshape(-1, 17, 1) * 60  # speed km/h
    # np_Y_MA = df_Y_MA.values.reshape(-1, 17, 1) * 360 / 60  # volume veh/min
    np_Y_MA = np_Y_MP_qk[:, :, 0].reshape(-1, 17, 1)  # density veh/min

    #######################
    np_X_MA[:, [2], :] = 0
    np_X_MA[:, [8], :] = 0

    np_X_MA = np_X_MA[:, [2,3,8], :]
    np_X_MP = np_X_MP[:, [2,3,8], :]
    np_X_MP_2 = np_X_MP_2[:, [2,3,8], :]
    np_Y_MA = np_Y_MA[:, [2,3,8], :]
    np_Y_MP = np_Y_MP[:, [2,3,8], :]
    edge_num = 3
    #######################
    
    
    # merge the y of MA and MP
    np_Y = np.concatenate((np_Y_MA, np_Y_MP), axis=2)

    # split the data into train validation and test 0.6 0.2 0.2
    # train
    X_MA_train = np_X_MA[:sum(df_day_time[:-4])]
    X_MP_train = np_X_MP[:sum(df_day_time[:-4])]
    X_MP_2_train = np_X_MP_2[:sum(df_day_time[:-4])]
    Y_train = np_Y[:sum(df_day_time[:-4])]
    # validation
    X_MA_val = np_X_MA[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    X_MP_val = np_X_MP[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    X_MP_2_val = np_X_MP_2[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    Y_val = np_Y[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    # test
    X_MA_test = np_X_MA[sum(df_day_time[:-3]):]
    X_MP_test = np_X_MP[sum(df_day_time[:-3]):]
    X_MP_2_test = np_X_MP_2[sum(df_day_time[:-3]):]
    Y_test = np_Y[sum(df_day_time[:-3]):]

    # # normalization  !!### here we try to only normalize the features ###!!
    # scaler_1, scaler_2, scaler_3 = StandardScaler(), StandardScaler(), StandardScaler()
    # X_MA_train[:, :, 0] = scaler_1.fit_transform(X_MA_train[:, :, 0])
    # X_MP_train[:, :, 0] = scaler_2.fit_transform(X_MP_train[:, :, 0])
    # X_MP_train[:, :, 1] = scaler_3.fit_transform(X_MP_train[:, :, 1])
    # X_MA_val[:, :, 0] = scaler_1.transform(X_MA_val[:, :, 0])
    # X_MP_val[:, :, 0] = scaler_2.transform(X_MP_val[:, :, 0])
    # X_MP_val[:, :, 1] = scaler_3.transform(X_MP_val[:, :, 1])
    # X_MA_test[:, :, 0] = scaler_1.transform(X_MA_test[:, :, 0])
    # X_MP_test[:, :, 0] = scaler_2.transform(X_MP_test[:, :, 0])
    # X_MP_test[:, :, 1] = scaler_3.transform(X_MP_test[:, :, 1])

    # normalization  !!### here we try to only normalize the features ###!!
    scaler_1, scaler_2, scaler_3 = StandardScaler(), StandardScaler(), StandardScaler()
    scaler_add1, scaler_add2 = StandardScaler(), StandardScaler()
    scaler_1.mean_, scaler_1.scale_, scaler_2.mean_, scaler_2.scale_, scaler_3.mean_, scaler_3.scale_, scaler_add1.mean_, scaler_add1.scale_, scaler_add2.mean_, scaler_add2.scale_ = scaler_x
    
    # scaler_num = StandardScaler()
    X_MA_train, X_MP_train = X_MA_train.reshape(-1, 1), X_MP_train.reshape(-1, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, 1), X_MP_val.reshape(-1, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, 1), X_MP_test.reshape(-1, 2)

    X_MP_2_train = X_MP_2_train.reshape(-1, num_features)
    X_MP_2_train[:, 0:1] = X_MP_2_train[:, 0:1]*1000
    X_MP_2_train[:, 1:2] = X_MP_2_train[:, 1:2]*60
    X_MP_2_val = X_MP_2_val.reshape(-1, num_features)
    X_MP_2_val[:, 0:1] = X_MP_2_val[:, 0:1]*1000
    X_MP_2_val[:, 1:2] = X_MP_2_val[:, 1:2]*60
    X_MP_2_test = X_MP_2_test.reshape(-1, num_features)
    X_MP_2_test[:, 0:1] = X_MP_2_test[:, 0:1]*1000
    X_MP_2_test[:, 1:2] = X_MP_2_test[:, 1:2]*60

    print(X_MA_train.shape, X_MP_train[:, 0:1].shape, X_MA_val.shape, X_MP_val.shape, X_MA_test.shape, X_MP_test.shape)
    
    X_MA_train[:, 0:1] = scaler_1.transform(X_MA_train[:, 0:1])
    X_MP_train[:, 0:1] = scaler_2.transform(X_MP_train[:, 0:1])
    X_MP_train[:, 1:2] = scaler_3.transform(X_MP_train[:, 1:2])
    X_MP_2_train[:, 0:1] = scaler_add1.transform(X_MP_2_train[:, 0:1]) ##
    X_MP_2_train[:, 1:2] = scaler_add2.transform(X_MP_2_train[:, 1:2]) ##
    # X_MP_train[:, 2:3] = scaler_num.fit_transform(X_MP_train[:, 2:3])
    X_MA_val[:, 0:1] = scaler_1.transform(X_MA_val[:, 0:1])
    X_MP_val[:, 0:1] = scaler_2.transform(X_MP_val[:, 0:1])
    X_MP_val[:, 1:2] = scaler_3.transform(X_MP_val[:, 1:2])
    X_MP_2_val[:, 0:1] = scaler_add1.transform(X_MP_2_val[:, 0:1]) ##
    X_MP_2_val[:, 1:2] = scaler_add2.transform(X_MP_2_val[:, 1:2]) ##
    # X_MP_val[:, 2:3] = scaler_num.transform(X_MP_val[:, 2:3])
    X_MA_test[:, 0:1] = scaler_1.transform(X_MA_test[:, 0:1])
    X_MP_test[:, 0:1] = scaler_2.transform(X_MP_test[:, 0:1])
    X_MP_test[:, 1:2] = scaler_3.transform(X_MP_test[:, 1:2])
    X_MP_2_test[:, 0:1] = scaler_add1.transform(X_MP_2_test[:, 0:1]) ##
    X_MP_2_test[:, 1:2] = scaler_add2.transform(X_MP_2_test[:, 1:2]) ##
    # X_MP_test[:, 2:3] = scaler_num.transform(X_MP_test[:, 2:3])
    X_MA_train, X_MP_train = X_MA_train.reshape(-1, edge_num, 1), X_MP_train.reshape(-1, edge_num, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, edge_num, 1), X_MP_val.reshape(-1, edge_num, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, edge_num, 1), X_MP_test.reshape(-1, edge_num, 2)
    X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_val.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_test.reshape(-1, edge_num, num_features*num_cells*num_steps)

    # normalization  !!### here we try to normalize the labels ###!!
    scaler_4, scaler_5 = StandardScaler(), StandardScaler()
    scaler_4.mean_, scaler_4.scale_, scaler_5.mean_, scaler_5.scale_ = scaler_y
    Y_train, Y_val, Y_test = Y_train.reshape(-1, 2), Y_val.reshape(-1, 2), Y_test.reshape(-1, 2)
    Y_train[:, 0:1] = scaler_4.transform(Y_train[:, 0:1])
    Y_train[:, 1:2] = scaler_5.transform(Y_train[:, 1:2])
    Y_val[:, 0:1] = scaler_4.transform(Y_val[:, 0:1])
    Y_val[:, 1:2] = scaler_5.transform(Y_val[:, 1:2])
    Y_test[:, 0:1] = scaler_4.transform(Y_test[:, 0:1])
    Y_test[:, 1:2] = scaler_5.transform(Y_test[:, 1:2])

    Y_train, Y_val, Y_test = Y_train.reshape(-1, edge_num, 2), Y_val.reshape(-1, edge_num, 2), Y_test.reshape(-1, edge_num, 2)

    test_data_real = get_time_series_pneuma(X_MA_test, X_MP_2_test, Y_test, df_day_time, 'TEST', 6, 1)

    return test_data_real

    # return X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, df_day_time, scaler
    # return X_MA_train, X_MP_2_train, Y_train, X_MA_val, X_MP_2_val, Y_val, X_MA_test, X_MP_2_test, Y_test, df_day_time, scaler

def get_time_series_pneuma(np_X_MA, np_X_MP, np_Y, df_day_time, mode, step_in, step_out):
    edge_num = 17
    # with the segment data we can get the time series data 
    # because the whole data is not continuous
    # start_val = sum(df_day_time[:-6])
    # start_test = sum(df_day_time[:-3])
    if mode == 'TRAIN':
        df_day_time = df_day_time[:-4]
        start = 0
    if mode == 'VALIDATION':
        df_day_time = df_day_time[-4:-3]
        start = 0
    if mode == 'TEST':
        df_day_time = df_day_time[-3:]
        start = 0
    # print("df_day_time", df_day_time)
    # print(np_X_MP.shape)

    XS_MA, XS_MP, YS = [], [], []
    for segment in df_day_time:
        for i in range(start, start + segment - step_out - step_in + 1):
            x_MA = np_X_MA[i:i + step_in, :, :]
            x_MP = np_X_MP[i:i + step_in, :, :]
            y = np_Y[i + step_in + step_out - 2, :, :]
            XS_MA.append(x_MA)
            XS_MP.append(x_MP)
            YS.append(y)
        start += segment
    XS_MA, XS_MP, YS = np.array(XS_MA), np.array(XS_MP), np.array(YS)
    YS = YS.reshape(-1, 1, 17, 2)
    XS_MA = XS_MA.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    XS_MP = XS_MP.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    YS = YS.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    print("XS_MA", XS_MA.shape, "XS_MP", XS_MP.shape, "YS", YS.shape)

    return XS_MA, XS_MP, YS



def load_data_sumo(MA_obs, PRMP, cell_rep):

    edge_num = 17 ######
    num_features = 2
    num_cells = 6
    num_steps = 2

    df_X_MA = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MA.csv")
    df_X_MP = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MP.csv")
    df_Y = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/label.csv")
    df_X_MP_2 = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MP_add.csv")

    df_X_MA = df_X_MA.drop(df_X_MA.columns[0], axis=1)
    df_X_MP = df_X_MP.drop(df_X_MP.columns[0], axis=1)
    df_Y = df_Y.drop(df_Y.columns[0], axis=1)
    df_X_MP_2 = df_X_MP_2.drop(df_X_MP_2.columns[0], axis=1)

    # MA features
    obs_list = [f'{edge}_{suffix}' for edge in MA_obs for suffix in ['C']]
    for column in df_X_MA.columns:
        if column not in obs_list:
            df_X_MA[column] = 0
    np_X_MA = df_X_MA.values.reshape(-1, edge_num, 1) # counts
    
    # MP features
    # one cell
    np_X_MP = df_X_MP.values.reshape(-1, edge_num, 3) # total travel time and total travel distance
    np_X_MP = np_X_MP[:, :, [0,1]] # delete the speed
    # multiple cells
    np_X_MP_2 = df_X_MP_2.values.reshape(-1, num_steps, 17, num_features*num_cells).transpose(0, 2, 1, 3).reshape(-1, 17, num_features*num_cells*num_steps) # speed
    
    # labels
    np_Y = df_Y.values.reshape(-1, 17, 3) # density veh/km and speed km/h
    np_Y = np_Y[:, :, 0:2]
    lane_1 = [10,12,13,15,16]
    lane_2 = [7,8,9,11,14]
    lane_4 = [3,4,5,6]
    lane_5 = [0,1,2]
    np_Y[:, lane_2, :] = np_Y[:, lane_2, :] / 2
    np_Y[:, lane_4, :] = np_Y[:, lane_4, :] / 4
    np_Y[:, lane_5, :] = np_Y[:, lane_5, :] / 5
    
    #######################
    mode = 'heavy' # 'light' or 'heavy'
    if mode == 'light':
        np_X_MA = np_X_MA[:, [2,3,8], :]
        np_X_MP = np_X_MP[:, [2,3,8], :]
        np_X_MP_2 = np_X_MP_2[:, [2,3,8], :]
        np_Y_MA = np_Y_MA[:, [2,3,8], :]
        np_Y = np_Y[:, [2,3,8], :]
        edge_num = 3
    else:
        edge_num = 17
    #######################
    
    # split the data according to length into train validation and test 0.6 0.2 0.2
    # train
    X_MA_train = np_X_MA[:len(np_X_MA)*6//10]
    X_MP_train = np_X_MP[:len(np_X_MP)*6//10]
    Y_train = np_Y[:len(np_Y)*6//10]
    X_MP_2_train = np_X_MP_2[:len(np_X_MP_2)*6//10]
    
    # validation
    X_MA_val = np_X_MA[len(np_X_MA)*6//10:len(np_X_MA)*7//10]
    X_MP_val = np_X_MP[len(np_X_MP)*6//10:len(np_X_MP)*7//10]
    Y_val = np_Y[len(np_Y)*6//10:len(np_Y)*7//10]
    X_MP_2_val = np_X_MP_2[len(np_X_MP_2)*6//10:len(np_X_MP_2)*7//10]
    
    # test
    X_MA_test = np_X_MA[len(np_X_MA)*7//10:]
    X_MP_test = np_X_MP[len(np_X_MP)*7//10:]
    Y_test = np_Y[len(np_Y)*7//10:]
    X_MP_2_test = np_X_MP_2[len(np_X_MP_2)*7//10:]

    # normalization  !!### here we try to only normalize the features ###!!
    scaler_1, scaler_2, scaler_3 = StandardScaler(), StandardScaler(), StandardScaler()
    X_MA_train, X_MP_train = X_MA_train.reshape(-1, 1), X_MP_train.reshape(-1, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, 1), X_MP_val.reshape(-1, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, 1), X_MP_test.reshape(-1, 2)
    scaler_add1, scaler_add2 = StandardScaler(), StandardScaler()
    X_MP_2_train = X_MP_2_train.reshape(-1, num_features)
    X_MP_2_val = X_MP_2_val.reshape(-1, num_features)
    X_MP_2_test = X_MP_2_test.reshape(-1, num_features)
    # X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train[:, 0:2], X_MP_2_val[:, 0:2], X_MP_2_test[:, 0:2]

    X_MA_train[:, 0:1] = scaler_1.fit_transform(X_MA_train[:, 0:1])
    X_MP_train[:, 0:1] = scaler_2.fit_transform(X_MP_train[:, 0:1])
    X_MP_train[:, 1:2] = scaler_3.fit_transform(X_MP_train[:, 1:2])
    X_MP_2_train[:, 0:1] = scaler_add1.fit_transform(X_MP_2_train[:, 0:1]) ##
    X_MP_2_train[:, 1:2] = scaler_add2.fit_transform(X_MP_2_train[:, 1:2]) ##

    X_MA_val[:, 0:1] = scaler_1.transform(X_MA_val[:, 0:1])
    X_MP_val[:, 0:1] = scaler_2.transform(X_MP_val[:, 0:1])
    X_MP_val[:, 1:2] = scaler_3.transform(X_MP_val[:, 1:2])
    X_MP_2_val[:, 0:1] = scaler_add1.transform(X_MP_2_val[:, 0:1]) ##
    X_MP_2_val[:, 1:2] = scaler_add2.transform(X_MP_2_val[:, 1:2]) ##

    X_MA_test[:, 0:1] = scaler_1.transform(X_MA_test[:, 0:1])
    X_MP_test[:, 0:1] = scaler_2.transform(X_MP_test[:, 0:1])
    X_MP_test[:, 1:2] = scaler_3.transform(X_MP_test[:, 1:2])
    X_MP_2_test[:, 0:1] = scaler_add1.transform(X_MP_2_test[:, 0:1]) ##
    X_MP_2_test[:, 1:2] = scaler_add2.transform(X_MP_2_test[:, 1:2]) ##

    X_MA_train, X_MP_train = X_MA_train.reshape(-1, edge_num, 1), X_MP_train.reshape(-1, edge_num, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, edge_num, 1), X_MP_val.reshape(-1, edge_num, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, edge_num, 1), X_MP_test.reshape(-1, edge_num, 2)
    X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_val.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_test.reshape(-1, edge_num, num_features*num_cells*num_steps)

    # normalization  !!### here we try to normalize the labels ###!!
    scaler_4, scaler_5 = StandardScaler(), StandardScaler()
    Y_train, Y_val, Y_test = Y_train.reshape(-1, 2), Y_val.reshape(-1, 2), Y_test.reshape(-1, 2)
    Y_train[:, 0:1] = scaler_4.fit_transform(Y_train[:, 0:1])
    Y_train[:, 1:2] = scaler_5.fit_transform(Y_train[:, 1:2])
    Y_val[:, 0:1] = scaler_4.transform(Y_val[:, 0:1])
    Y_val[:, 1:2] = scaler_5.transform(Y_val[:, 1:2])
    Y_test[:, 0:1] = scaler_4.transform(Y_test[:, 0:1])
    Y_test[:, 1:2] = scaler_5.transform(Y_test[:, 1:2])


    Y_train, Y_val, Y_test = Y_train.reshape(-1, edge_num, 2), Y_val.reshape(-1, edge_num, 2), Y_test.reshape(-1, edge_num, 2)
    scaler = scaler_4.mean_, scaler_4.scale_, scaler_5.mean_, scaler_5.scale_


    print(X_MA_train.shape, X_MP_2_train.shape, Y_train.shape, X_MA_val.shape, X_MP_2_val.shape, Y_val.shape, X_MA_test.shape, X_MP_2_test.shape, Y_test.shape)
    
    if cell_rep:
        return X_MA_train, X_MP_2_train, Y_train, X_MA_val, X_MP_2_val, Y_val, X_MA_test, X_MP_2_test, Y_test, scaler
    else:
        return X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, scaler

def get_time_series_sumo(np_X_MA, np_X_MP, np_Y, step_in, step_out):

    start = 0
    XS_MA, XS_MP, YS = [], [], []
    for i in range(start, len(np_X_MA) - step_out - step_in + 1):
        x_MA = np_X_MA[i:i + step_in, :, :]
        x_MP = np_X_MP[i:i + step_in, :, :]
        y = np_Y[i + step_in + step_out - 2, :, :]
        XS_MA.append(x_MA)
        XS_MP.append(x_MP)
        YS.append(y)
    # print(len(XS_MA), len(XS_MP), len(YS))
    # print(XS_MA[0].shape, XS_MP[0].shape, YS[0].shape)
    XS_MA, XS_MP, YS = np.array(XS_MA), np.array(XS_MP), np.array(YS)
    YS = YS.reshape(-1, 1, 17, 2)
    XS_MA = XS_MA.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    XS_MP = XS_MP.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    YS = YS.transpose(0, 3, 1, 2)  # [B,T,N,C] -> [B,C,T,N]
    print("XS_MA", XS_MA.shape, "XS_MP", XS_MP.shape, "YS", YS.shape)

    return XS_MA, XS_MP, YS




def load_data_pneuma(MA_obs, PRMP, target_edges, cell_rep):
    df_list_MA = []
    df_list_MP = []
    df_list_MP_add = []
    df_list_label = []
    df_day_time = []

    edge_num = 17 ######
    num_features = 2
    num_cells = 6
    num_steps = 2

    days = ["20181024", "20181029", "20181030","20181101"]
    times = ["0800_0830", "0830_0900", "0900_0930", "0930_1000", "1000_1030", "1030_1100"]

    for day in days:
        for timetime in times:
            # some data is not good to use
            if not os.path.exists(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MA.csv"):
                continue
            if day == "20181101" and timetime == "0800_0830":
                continue
            if day == "20181029" and timetime == "1000_1030":
                continue
            if day == "20181030" and timetime == "0930_1000":
                continue
            if day == "20181030" and timetime == "1000_1030":
                continue
            # read data
            df_MA = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MA.csv")
            df_list_MA.append(df_MA)
            df_MP = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MP.csv")
            df_list_MP.append(df_MP)
            df_MP_add = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_MP_add.csv")
            df_list_MP_add.append(df_MP_add)
            df_label = pd.read_csv(f"../datasets/pneuma_corridor/MP_{PRMP}_ma/{day}_{timetime}_label.csv")
            df_list_label.append(df_label)
            df_day_time.append(len(df_MA))

    df_X_MA = pd.concat(df_list_MA)
    df_X_MP = pd.concat(df_list_MP)
    df_X_MP_2 = pd.concat(df_list_MP_add)
    df_Y = pd.concat(df_list_label)
    df_X_MA = df_X_MA.drop(df_X_MA.columns[0], axis=1)
    df_X_MP = df_X_MP.drop(df_X_MP.columns[0], axis=1)
    df_X_MP_2 = df_X_MP_2.drop(df_X_MP_2.columns[0], axis=1)
    df_Y = df_Y.drop(df_Y.columns[0], axis=1)

    # MA features
    obs_list = [f'{edge}_{suffix}' for edge in MA_obs for suffix in ['e']]
    for column in df_X_MA.columns:
        if column not in obs_list:
            df_X_MA[column] = 0
    np_X_MA = df_X_MA.values.reshape(-1, edge_num, 2) # counts
    np_X_MA = np_X_MA[:, :, 1:2] # delete the input traffic counts

    # MP features
    # one cell
    np_X_MP = df_X_MP.values.reshape(-1, edge_num, 3) # total travel time and total travel distance
    np_X_MP = np_X_MP[:, :, [0,1]] # delete the speed
    # multiple cells
    np_X_MP_2 = df_X_MP_2.values.reshape(-1, num_steps, edge_num, num_features*num_cells).transpose(0, 2, 1, 3).reshape(-1, edge_num, num_features*num_cells*num_steps)
    
    # labels
    # target_list = [f'{edge}_{suffix}' for edge in target_edges for suffix in ['D', 'T', 'V']]
    # for column in df_Y.columns:
    #     if column not in target_list:
    #         df_Y = df_Y.drop(column, axis=1)
    np_Y = df_Y.values.reshape(-1, 17, 3)
    np_Y = np_Y[:, :, 0:2]
    lane_1 = [10,12,13,15,16]
    lane_2 = [7,8,9,11,14]
    lane_4 = [3,4,5,6]
    lane_5 = [0,1,2]
    np_Y[:, :, lane_2] = np_Y[:, :, lane_2] / 2
    np_Y[:, :, lane_4] = np_Y[:, :, lane_4] / 4
    np_Y[:, :, lane_5] = np_Y[:, :, lane_5] / 5
    
        

    #########################################
    mode = 'heavy' # 'light' or 'heavy'
    if mode == 'light':
        np_X_MA = np_X_MA[:, [2,3,8], :]
        np_X_MP = np_X_MP[:, [2,3,8], :]
        np_X_MP_2 = np_X_MP_2[:, [2,3,8], :]
        np_Y_MA = np_Y_MA[:, [2,3,8], :]
        np_Y = np_Y[:, [2,3,8], :]
        edge_num = 3
    else:
        edge_num = 17
    #########################################
    
    # split the data into train validation and test
    # train
    X_MA_train = np_X_MA[:sum(df_day_time[:-4])]
    X_MP_train = np_X_MP[:sum(df_day_time[:-4])]
    X_MP_2_train = np_X_MP_2[:sum(df_day_time[:-4])]
    Y_train = np_Y[:sum(df_day_time[:-4])]
    # validation
    X_MA_val = np_X_MA[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    X_MP_val = np_X_MP[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    X_MP_2_val = np_X_MP_2[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    Y_val = np_Y[sum(df_day_time[:-4]):sum(df_day_time[:-3])]
    # test
    X_MA_test = np_X_MA[sum(df_day_time[:-3]):]
    X_MP_test = np_X_MP[sum(df_day_time[:-3]):]
    X_MP_2_test = np_X_MP_2[sum(df_day_time[:-3]):]
    Y_test = np_Y[sum(df_day_time[:-3]):]


    # normalization  !!### here we try to only normalize the features ###!!
    scaler_1, scaler_2, scaler_3 = StandardScaler(), StandardScaler(), StandardScaler()
    scaler_add1, scaler_add2 = StandardScaler(), StandardScaler()
    X_MA_train, X_MP_train = X_MA_train.reshape(-1, 1), X_MP_train.reshape(-1, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, 1), X_MP_val.reshape(-1, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, 1), X_MP_test.reshape(-1, 2)

    X_MP_2_train = X_MP_2_train.reshape(-1, num_features)
    X_MP_2_val = X_MP_2_val.reshape(-1, num_features)
    X_MP_2_test = X_MP_2_test.reshape(-1, num_features)

    X_MA_train[:, 0:1] = scaler_1.fit_transform(X_MA_train[:, 0:1])
    X_MP_train[:, 0:1] = scaler_2.fit_transform(X_MP_train[:, 0:1])
    X_MP_train[:, 1:2] = scaler_3.fit_transform(X_MP_train[:, 1:2])
    X_MP_2_train[:, 0:1] = scaler_add1.fit_transform(X_MP_2_train[:, 0:1]) ##
    X_MP_2_train[:, 1:2] = scaler_add2.fit_transform(X_MP_2_train[:, 1:2]) ##
    # X_MP_train[:, 2:3] = scaler_num.fit_transform(X_MP_train[:, 2:3])
    X_MA_val[:, 0:1] = scaler_1.transform(X_MA_val[:, 0:1])
    X_MP_val[:, 0:1] = scaler_2.transform(X_MP_val[:, 0:1])
    X_MP_val[:, 1:2] = scaler_3.transform(X_MP_val[:, 1:2])
    X_MP_2_val[:, 0:1] = scaler_add1.transform(X_MP_2_val[:, 0:1]) ##
    X_MP_2_val[:, 1:2] = scaler_add2.transform(X_MP_2_val[:, 1:2]) ##
    # X_MP_val[:, 2:3] = scaler_num.transform(X_MP_val[:, 2:3])
    X_MA_test[:, 0:1] = scaler_1.transform(X_MA_test[:, 0:1])
    X_MP_test[:, 0:1] = scaler_2.transform(X_MP_test[:, 0:1])
    X_MP_test[:, 1:2] = scaler_3.transform(X_MP_test[:, 1:2])
    X_MP_2_test[:, 0:1] = scaler_add1.transform(X_MP_2_test[:, 0:1]) ##
    X_MP_2_test[:, 1:2] = scaler_add2.transform(X_MP_2_test[:, 1:2]) ##
    # X_MP_test[:, 2:3] = scaler_num.transform(X_MP_test[:, 2:3])
    X_MA_train, X_MP_train = X_MA_train.reshape(-1, edge_num, 1), X_MP_train.reshape(-1, edge_num, 2)
    X_MA_val, X_MP_val = X_MA_val.reshape(-1, edge_num, 1), X_MP_val.reshape(-1, edge_num, 2)
    X_MA_test, X_MP_test = X_MA_test.reshape(-1, edge_num, 1), X_MP_test.reshape(-1, edge_num, 2)
    X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_val.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_test.reshape(-1, edge_num, num_features*num_cells*num_steps)

    # normalization  !!### here we try to normalize the labels ###!!
    scaler_4, scaler_5 = StandardScaler(), StandardScaler()
    Y_train, Y_val, Y_test = Y_train.reshape(-1, 2), Y_val.reshape(-1, 2), Y_test.reshape(-1, 2)
    Y_train[:, 0:1] = scaler_4.fit_transform(Y_train[:, 0:1])
    Y_train[:, 1:2] = scaler_5.fit_transform(Y_train[:, 1:2])
    Y_val[:, 0:1] = scaler_4.transform(Y_val[:, 0:1])
    Y_val[:, 1:2] = scaler_5.transform(Y_val[:, 1:2])
    Y_test[:, 0:1] = scaler_4.transform(Y_test[:, 0:1])
    Y_test[:, 1:2] = scaler_5.transform(Y_test[:, 1:2])

    Y_train, Y_val, Y_test = Y_train.reshape(-1, 17, 2), Y_val.reshape(-1, 17, 2), Y_test.reshape(-1, 17, 2)
    scaler = scaler_4.mean_, scaler_4.scale_, scaler_5.mean_, scaler_5.scale_

    if cell_rep:
        return X_MA_train, X_MP_2_train, Y_train, X_MA_val, X_MP_2_val, Y_val, X_MA_test, X_MP_2_test, Y_test, df_day_time, scaler
    else:
        return X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, df_day_time, scaler



# def load_data_sumo_pneuma(MA_obs, PRMP):

#     edge_num = 17 ######
#     num_features = 2
#     num_cells = 6
#     num_steps = 2

#     df_X_MA = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MA.csv")
#     df_X_MP = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MP.csv")
#     df_Y = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/label.csv")
#     df_X_MP_2 = pd.read_csv(f"../datasets/sumo_corridor/MP_{PRMP}_ma/MP_add.csv")

#     df_X_MA = df_X_MA.drop(df_X_MA.columns[0], axis=1)
#     df_X_MP = df_X_MP.drop(df_X_MP.columns[0], axis=1)
#     df_Y = df_Y.drop(df_Y.columns[0], axis=1)
#     df_X_MP_2 = df_X_MP_2.drop(df_X_MP_2.columns[0], axis=1)

#     # MA features
#     obs_list = [f'{edge}_{suffix}' for edge in MA_obs for suffix in ['C']]
#     for column in df_X_MA.columns:
#         if column not in obs_list:
#             df_X_MA[column] = 0
#     np_X_MA = df_X_MA.values.reshape(-1, edge_num, 1) # counts
    
#     # MP features
#     # one cell
#     np_X_MP = df_X_MP.values.reshape(-1, edge_num, 3) # total travel time and total travel distance
#     np_X_MP = np_X_MP[:, :, [0,1]] # delete the speed
#     # multiple cells
#     np_X_MP_2 = df_X_MP_2.values.reshape(-1, num_steps, 17, num_features*num_cells).transpose(0, 2, 1, 3).reshape(-1, 17, num_features*num_cells*num_steps) # speed
    
#     # labels
#     np_Y = df_Y.values.reshape(-1, 17, 3) # density veh/km and speed km/h
#     np_Y = np_Y[:, :, 0:2]
    
#     #######################
#     mode = 'heavy' # 'light' or 'heavy'
#     if mode == 'light':
#         np_X_MA = np_X_MA[:, [2,3,8], :]
#         np_X_MP = np_X_MP[:, [2,3,8], :]
#         np_X_MP_2 = np_X_MP_2[:, [2,3,8], :]
#         np_Y_MA = np_Y_MA[:, [2,3,8], :]
#         np_Y = np_Y[:, [2,3,8], :]
#         edge_num = 3
#     else:
#         edge_num = 17
#     #######################
    
#     # split the data according to length into train validation and test 0.6 0.2 0.2
#     # train
#     X_MA_train = np_X_MA[:len(np_X_MA)*6//10]
#     X_MP_train = np_X_MP[:len(np_X_MP)*6//10]
#     Y_train = np_Y[:len(np_Y)*6//10]
#     X_MP_2_train = np_X_MP_2[:len(np_X_MP_2)*6//10]
    
#     # validation
#     X_MA_val = np_X_MA[len(np_X_MA)*6//10:len(np_X_MA)*7//10]
#     X_MP_val = np_X_MP[len(np_X_MP)*6//10:len(np_X_MP)*7//10]
#     Y_val = np_Y[len(np_Y)*6//10:len(np_Y)*7//10]
#     X_MP_2_val = np_X_MP_2[len(np_X_MP_2)*6//10:len(np_X_MP_2)*7//10]
    
#     # test
#     X_MA_test = np_X_MA[len(np_X_MA)*7//10:]
#     X_MP_test = np_X_MP[len(np_X_MP)*7//10:]
#     Y_test = np_Y[len(np_Y)*7//10:]
#     X_MP_2_test = np_X_MP_2[len(np_X_MP_2)*7//10:]

#     # normalization  !!### here we try to only normalize the features ###!!
#     scaler_1, scaler_2, scaler_3 = StandardScaler(), StandardScaler(), StandardScaler()
#     X_MA_train, X_MP_train = X_MA_train.reshape(-1, 1), X_MP_train.reshape(-1, 2)
#     X_MA_val, X_MP_val = X_MA_val.reshape(-1, 1), X_MP_val.reshape(-1, 2)
#     X_MA_test, X_MP_test = X_MA_test.reshape(-1, 1), X_MP_test.reshape(-1, 2)
#     scaler_add1, scaler_add2 = StandardScaler(), StandardScaler()
#     X_MP_2_train = X_MP_2_train.reshape(-1, num_features)
#     X_MP_2_val = X_MP_2_val.reshape(-1, num_features)
#     X_MP_2_test = X_MP_2_test.reshape(-1, num_features)
#     # X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train[:, 0:2], X_MP_2_val[:, 0:2], X_MP_2_test[:, 0:2]

#     X_MA_train[:, 0:1] = scaler_1.fit_transform(X_MA_train[:, 0:1])
#     X_MP_train[:, 0:1] = scaler_2.fit_transform(X_MP_train[:, 0:1])
#     X_MP_train[:, 1:2] = scaler_3.fit_transform(X_MP_train[:, 1:2])
#     X_MP_2_train[:, 0:1] = scaler_add1.fit_transform(X_MP_2_train[:, 0:1]) ##
#     X_MP_2_train[:, 1:2] = scaler_add2.fit_transform(X_MP_2_train[:, 1:2]) ##

#     X_MA_val[:, 0:1] = scaler_1.transform(X_MA_val[:, 0:1])
#     X_MP_val[:, 0:1] = scaler_2.transform(X_MP_val[:, 0:1])
#     X_MP_val[:, 1:2] = scaler_3.transform(X_MP_val[:, 1:2])
#     X_MP_2_val[:, 0:1] = scaler_add1.transform(X_MP_2_val[:, 0:1]) ##
#     X_MP_2_val[:, 1:2] = scaler_add2.transform(X_MP_2_val[:, 1:2]) ##

#     X_MA_test[:, 0:1] = scaler_1.transform(X_MA_test[:, 0:1])
#     X_MP_test[:, 0:1] = scaler_2.transform(X_MP_test[:, 0:1])
#     X_MP_test[:, 1:2] = scaler_3.transform(X_MP_test[:, 1:2])
#     X_MP_2_test[:, 0:1] = scaler_add1.transform(X_MP_2_test[:, 0:1]) ##
#     X_MP_2_test[:, 1:2] = scaler_add2.transform(X_MP_2_test[:, 1:2]) ##

#     X_MA_train, X_MP_train = X_MA_train.reshape(-1, edge_num, 1), X_MP_train.reshape(-1, edge_num, 2)
#     X_MA_val, X_MP_val = X_MA_val.reshape(-1, edge_num, 1), X_MP_val.reshape(-1, edge_num, 2)
#     X_MA_test, X_MP_test = X_MA_test.reshape(-1, edge_num, 1), X_MP_test.reshape(-1, edge_num, 2)
#     X_MP_2_train, X_MP_2_val, X_MP_2_test = X_MP_2_train.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_val.reshape(-1, edge_num, num_features*num_cells*num_steps), X_MP_2_test.reshape(-1, edge_num, num_features*num_cells*num_steps)

#     # normalization  !!### here we try to normalize the labels ###!!
#     scaler_4, scaler_5 = StandardScaler(), StandardScaler()
#     Y_train, Y_val, Y_test = Y_train.reshape(-1, 2), Y_val.reshape(-1, 2), Y_test.reshape(-1, 2)
#     Y_train[:, 0:1] = scaler_4.fit_transform(Y_train[:, 0:1])
#     Y_train[:, 1:2] = scaler_5.fit_transform(Y_train[:, 1:2])
#     Y_val[:, 0:1] = scaler_4.transform(Y_val[:, 0:1])
#     Y_val[:, 1:2] = scaler_5.transform(Y_val[:, 1:2])
#     Y_test[:, 0:1] = scaler_4.transform(Y_test[:, 0:1])
#     Y_test[:, 1:2] = scaler_5.transform(Y_test[:, 1:2])



#     Y_train, Y_val, Y_test = Y_train.reshape(-1, edge_num, 2), Y_val.reshape(-1, edge_num, 2), Y_test.reshape(-1, edge_num, 2)
    
#     scaler_x = scaler_1.mean_, scaler_1.scale_, scaler_2.mean_, scaler_2.scale_, scaler_3.mean_, scaler_3.scale_, scaler_add1.mean_, scaler_add1.scale_, scaler_add2.mean_, scaler_add2.scale_
#     scaler_y = scaler_4.mean_, scaler_4.scale_, scaler_5.mean_, scaler_5.scale_

#     # test data from real world
#     test_data_real = load_data_pneuma_test(PRMA, PRMP, scaler_x, scaler_y)

#     print(X_MA_train.shape, X_MP_2_train.shape, Y_train.shape, X_MA_val.shape, X_MP_2_val.shape, Y_val.shape, X_MA_test.shape, X_MP_2_test.shape, Y_test.shape)
    
#     # return X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, scaler
#     return X_MA_train, X_MP_2_train, Y_train, X_MA_val, X_MP_2_val, Y_val, X_MA_test, X_MP_2_test, Y_test, scaler_y, test_data_real
