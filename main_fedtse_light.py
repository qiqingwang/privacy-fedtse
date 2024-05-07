import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
from pytorch_stgcn import *
import argparse
from data_process_pneuma import load_data, get_time_series
from pytorch_mlp import FusionModel, FusionModel_test
from pytorch_stgcn import *
from fedtse import VFLHostMA, VFLGuestMP, FedTSE
from fedtse_fixture import FederatedLearningFixture
from utils import find_your_path

'''
split the data into train and test ? ##### finished #####
normalize the data ? For this I will try to not normalize the label only the features. ##### finished #####
learning rate decay ?
save the model ?
evaluate the model ? need to be careful ! ##### finished ##### where we do not normalize the label
save the necessary results for plotting ?  ##### finished #####
define all the parameters in one place ?
'''

parser = argparse.ArgumentParser()
# parser.add_argument("--", type=, default=, help="")
parser.add_argument("--dataname", type=str, default="MA_1_MP_1.0", help="dataset name")
parser.add_argument("--timestep_in",type=int,default=12,help="the time step you input")
parser.add_argument("--timestep_out", type=int, default=1, help="the time step will output")
parser.add_argument("--n_node", type=int, default=17, help="the number of the node")
parser.add_argument("--channel", type=int, default=2, help="number of channel")
parser.add_argument("--batchsize", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
# parser.add_argument("--lr_g", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--epoch", type=int, default=500, help="number of epochs of training")
parser.add_argument("--patience", type=float, default=100, help="patience used for early stop")
parser.add_argument("--optimizer", type=str, default='Adam', help="RMSprop, Adam, SGD")
parser.add_argument("--loss", type=str, default='MSE', help="MAE, MSE, SELF")
parser.add_argument("--trainvalsplit", type=float, default=0.1919, help="val_ratio = 0.8 * 0.125 = 0.1") # val_ratio = 0.8 * 0.125 = 0.1
parser.add_argument("--adjpath", type=str, default='../datasets_pneuma_corrider/adjacency_matrix.csv', help="the absolute path of adj file")
parser.add_argument("--adjtype", type=str, default="symnadj", help="the type of adj")
parser.add_argument('--ex', type=str, default='typhoon-inflow', help='which experiment setting to run')
parser.add_argument('--gpu',type=int,default=0,help='gpu num')
parser.add_argument('--target',type=int,default=0,help='predict target 0:bike inflow 1: bike outflow 2:taxi inflow 3:taxi outflow')
parser.add_argument('--multigraph', type=bool, default=False, help="Is multi-graph prediction")
parser.add_argument('--datatype',type=int,default=0,help='the type of data, 0 is taxi, 1 is bike')
parser.add_argument('--addtime', type=bool, default=False, help="Add timestamp")
parser.add_argument('--PRMA', type=float, default=1, help="PRMA")
parser.add_argument('--PRMP', type=float, default=1.0, help="PRMP")
# parser.add_argument('--communication', type=int, default=1, help="communication")
opt = parser.parse_args()


DATANAME = opt.dataname
TIMESTEP_OUT = opt.timestep_out
TIMESTEP_IN = opt.timestep_in
N_NODE = opt.n_node
CHANNEL = opt.channel
BATCHSIZE = opt.batchsize
# LEARN = opt.lr
# LEARN_g = opt.lr_g
EPOCH = opt.epoch
PATIENCE = opt.patience
OPTIMIZER = opt.optimizer
LOSS = opt.loss
TRAINVALSPLIT = opt.trainvalsplit
ADJPATH = opt.adjpath
ADJTYPE = opt.adjtype
GPU = opt.gpu
TARGET = opt.target
GRAPHNUMBER = 1 + int(opt.multigraph)
DATATYPE = opt.datatype
ADDTIME = opt.addtime
# MULTIGRAPH = opt.multigraph
PRMA = opt.PRMA
PRMP = opt.PRMP
# Q = opt.q

# check the parameters settings in STGCN !
def getModel(num, MULTIGRAPH, learning_rate):
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    GRAPHNUMBER = 1 + int(MULTIGRAPH)
    ks, kt, bs, T, n, p = 3, 3, [[num, 32, 64], [64, 32, 64]], TIMESTEP_IN, N_NODE, 0
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
    model = STGCN(ks, kt, bs, T, n, Lk_new, p, learning_rate).to(device)
    return model

def run_experiment(train_data, val_data, test_data, batch_size, learning_rate, epoch):
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
    # party_a_local_model = getModel(1, False, learning_rate)
    # party_b_local_model = getModel(2, False, learning_rate)
    party_a_local_model = FusionModel_test(input_size=1 * 17 * 12, hidden_size=200 * 2 * 17,
                                      output_size=1 * 2 * 17, lr=learning_rate).to(device)
    party_b_local_model = FusionModel_test(input_size=2 * 17 * 12, hidden_size=200 * 2 * 17,
                                      output_size=1 * 2 * 17, lr=learning_rate).to(device)
    party_a_global_model = FusionModel(input_size=2 * 2 * 17, hidden_size=200 * 2 * 17, 
                                       output_size=1 * 2 * 17, lr=learning_rate).to(device)
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
    fl_fixture.fit(train_data=train_data, val_data=val_data, test_data=test_data, epochs=epoch, batch_size=batch_size, PATH=PATH)

if __name__ == '__main__':

    find_your_path()
    # set the seed
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    # set the device
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

    Q = 1
    LEARN = 0.0001
    # set the keyword for saving the results
    KEYWORD = 'FedTSE_pneuma' + '_' + f'MA_{PRMA}_MP_{PRMP}' + f'_{LEARN}_{Q}' + '_' + datetime.now().strftime("%y%m%d%H%M")
    PATH = '../save/' + KEYWORD
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    print("############################### Prepare Data #############################")
    # load data
    X_MA_train, X_MP_train, Y_train, X_MA_val, X_MP_val, Y_val, X_MA_test, X_MP_test, Y_test, df_day_time = load_data(PRMA, PRMP)
    # process to time series
    train_data = get_time_series(X_MA_train, X_MP_train, Y_train, df_day_time, 'TRAIN', TIMESTEP_IN, TIMESTEP_OUT)
    val_data = get_time_series(X_MA_val, X_MP_val, Y_val, df_day_time, 'VALIDATION', TIMESTEP_IN, TIMESTEP_OUT)
    test_data = get_time_series(X_MA_test, X_MP_test, Y_test, df_day_time, 'TEST', TIMESTEP_IN, TIMESTEP_OUT)

    print("############################# Experiment start ###########################")
    run_experiment(train_data=train_data, val_data=val_data, test_data=test_data, batch_size=BATCHSIZE, learning_rate=LEARN, epoch=EPOCH)


