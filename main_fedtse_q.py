import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from src.data_process import load_data_pneuma, get_time_series_pneuma, load_data_sumo, get_time_series_sumo
from src.mlp import FusionModel
from src.stgcn_q import *
from src.fedtse_party import VFLHostMA, VFLGuestMP, FedTSE
from src.fedtse_fixture import FederatedLearningFixture
from src.utils import find_your_path
import shutil

# check the parameters settings in STGCN !
class conv(nn.Module):
    def __init__(self, LEARN, Q):
        super(conv, self).__init__()
        # self.conv1 = nn.Conv2d(2, 16, (1, 1))
        # self.deconv1 = nn.ConvTranspose2d(2, 2, (1, 3))
        # self.conv2 = nn.Conv2d(4, 2, (1, 1))
        # self.unpooling = nn.MaxUnpool2d((1, 5), 1)
        # self.conv3 = nn.Conv2d(8, 2, (1, 1))

        self.fc1 = nn.Linear(2*9, 16*17)
        self.fc2 = nn.Linear(16*17, 2*17)
        # self.fc3 = nn.Linear(2*14, 2*17)
        self.relu = nn.ReLU()
        
        # self.pooling = nn.MaxPool2d(2, 2)
        # self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARN, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=140, gamma=0.99)
        self.Q = Q
    
    def forward(self, x):
        x = x.reshape(-1, 2*9)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(-1, 2, 1, 17)
        return x
    
    def backward(self, x, grads):
        self.optimizer.zero_grad()
        output = self.forward(x) 
        output.backward(grads)
        for i in range(self.Q):
            self.optimizer.step()
        self.scheduler.step()
        x_grad = x.grad
        return x_grad


def get_local_model(num, MULTIGRAPH, learning_rate, ADJPATH, o, mode, Q):
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    GRAPHNUMBER = 1 + int(MULTIGRAPH)
    ks, kt, bs, T, n, p = 3, 3, [[num, 16, 64], [64, 16, 64]], TIMESTEP_IN, 17, 0
    Lk_new = torch.empty(GRAPHNUMBER,3,17,17).to(device)
    adjpathlist = []
    adjpathlist.append(ADJPATH)
    if MULTIGRAPH:
        adjpathlist.append('../datasets_pneuma/OD_matrix.csv')
    for i in range(GRAPHNUMBER):
        A = pd.read_csv(adjpathlist[i]).values
        W = weight_matrix(A)
        L = scaled_laplacian(W)
        # print('L shape is', L.shape)
        Lk = cheb_poly(L, ks)
        # print('cheb_poly', Lk.shape)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        Lk_new[i] = Lk
    # print(Lk_new.shape)
    model = STGCN(ks, kt, bs, T, n, Lk_new, p, learning_rate, o, mode, Q).to(device)
    return model

def run_experiment(train_data, val_data, test_data, batch_size, learning_rate, epoch, adj_matrix):
    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))

    trainXS_host, trainXS_guest, trainYS_host = train_data
    valXS_host, valXS_guest, valYS_host = val_data
    testXS_host, testXS_guest, testYS_host = test_data
    # to tensor and to device
    trainXS_host, trainXS_guest, trainYS_host = torch.Tensor(trainXS_host).to(device), torch.Tensor(trainXS_guest).to(device), torch.Tensor(trainYS_host).to(device)
    valXS_host, valXS_guest = torch.Tensor(valXS_host).to(device), torch.Tensor(valXS_guest).to(device)
    testXS_host, testXS_guest = torch.Tensor(testXS_host).to(device), torch.Tensor(testXS_guest).to(device)

    print("############# Wire Federated Models ############")
    # create local models and global model for both party A and party B
    party_a_local_model = get_local_model(1, False, learning_rate, adj_matrix, 2, mode = 'vfl', Q = Q)
    party_b_local_model = get_local_model(2, False, learning_rate, adj_matrix, 2, mode = 'vfl', Q = Q)
    party_a_global_model = conv(learning_rate, Q).to(device)

    # party_a_global_model = FusionModel(input_size=2 * 2 * 17, hidden_size=200 * 2 * 17, 
    #                                    output_size=1 * 2 * 17, lr=learning_rate).to(device)
    
    # assign models to both party A and party B
    partyA = VFLHostMA(local_model=party_a_local_model, global_model=party_a_global_model)
    partyB = VFLGuestMP(local_model=party_b_local_model)
    # add party B to party A's party list
    party_B_id = "B"
    federatedLearning = FedTSE(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.set_debug(is_debug=False)

    print("############# Train Federated Models ############")
    fl_fixture = FederatedLearningFixture(federatedLearning)

    # only party A has labels (i.e., Y), other parties only have features (e.g., X).
    # 'party_list' stores X for all other parties.
    # Since this is two-party VFL, 'party_list' only stores the X of party B.
    train_data = {federatedLearning.get_main_party_id(): {"X": trainXS_host, "Y": trainYS_host},
                  "party_list": {party_B_id: trainXS_guest}}
    val_data = {federatedLearning.get_main_party_id(): {"X": valXS_host, "Y": valYS_host},
                "party_list": {party_B_id: valXS_guest}}
    test_data = {federatedLearning.get_main_party_id(): {"X": testXS_host, "Y": testYS_host},
                 "party_list": {party_B_id: testXS_guest}}
    fl_fixture.fit(train_data=train_data, val_data=val_data, test_data=test_data, epochs=epoch, batch_size=batch_size, PATH=PATH, scaler=scaler)

if __name__ == '__main__':
    for LEARN in [0.0001, 0.0002, 0.00015, 0.0004, 0.00007]:
        Q = 2
        PRMP = 0.2
        find_your_path()
        # set the seed
        torch.manual_seed(2023)
        torch.cuda.manual_seed(2023)
        np.random.seed(2023)
        # set the device
        device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device("cpu")

        # set the training parameters
        # LEARN = 0.0003 # 0.0001
        TIMESTEP_OUT = 1 # 1
        TIMESTEP_IN = 9 # 6
        BATCHSIZE = 128
        EPOCH = 2000 # 500
        MA_obs = ['E1', 'E5']
        # PRMP = 0.5
        train_mode = 'simulation' # 'simulation' or 'real'
        cell_rep = False

        # set the graph parameters
        # the traj of on_R5 on_R2 on_R6 off_R4  is not recorded by the drones in pNEUMA datasets
        # input_edges = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'off_R1', 'off_R2', 'off_R3', 'on_R1', 'on_R3', 'on_R4']
        target_edges = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
        adj_matrix = '../datasets/adjacency_matrix.csv'
        MA_obs = ['E1', 'E5']

        # set the keyword for saving the results
        KEYWORD = 'fedtse' + '_' + f'MP_{Q}' + f'_{LEARN}' + '_' + datetime.now().strftime("%y%m%d%H%M")
        PATH = '../save_sumo_fedtse_Q/' + KEYWORD
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        currentPython = sys.argv[0]
        shutil.copy2(currentPython, PATH)
        
    
        print("############################### Prepare Data #############################")
        
        if train_mode == 'simulation':
            # load data
            X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, scaler = load_data_sumo(MA_obs, PRMP, cell_rep)
            # process to time series
            train_data = get_time_series_sumo(X_MA_train, X_MP_train, Y_train, TIMESTEP_IN, TIMESTEP_OUT)
            val_data = get_time_series_sumo(X_MA_val, X_MP_val, Y_val, TIMESTEP_IN, TIMESTEP_OUT)
            test_data = get_time_series_sumo(X_MA_test, X_MP_test, Y_test, TIMESTEP_IN, TIMESTEP_OUT)
        else:
            # load data
            X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, df_day_time, scaler = load_data_pneuma(MA_obs, PRMP, target_edges, cell_rep)
            # process to time series
            train_data = get_time_series_pneuma(X_MA_train, X_MP_train, Y_train, df_day_time, 'TRAIN', TIMESTEP_IN, TIMESTEP_OUT)
            val_data = get_time_series_pneuma(X_MA_val, X_MP_val, Y_val, df_day_time, 'VALIDATION', TIMESTEP_IN, TIMESTEP_OUT)
            test_data = get_time_series_pneuma(X_MA_test, X_MP_test, Y_test, df_day_time, 'TEST', TIMESTEP_IN, TIMESTEP_OUT)

        print("############################# Experiment start ###########################")
        run_experiment(train_data=train_data, val_data=val_data, test_data=test_data,
                    batch_size=BATCHSIZE, learning_rate=LEARN, epoch=EPOCH,
                    adj_matrix=adj_matrix)