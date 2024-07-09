import os
import sys
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from src.data_process import load_data
from src.mlp import MLP
from src.stgcn import *
from src.fedtse_party import VFLHostMA, VFLGuestMP, FedTSE
from src.fedtse_fixture import FederatedLearningFixture


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
    party_a_local_model = get_local_model(1, False, learning_rate, adj_matrix, 2, args.node_num, args.ReducedD, args.Q, args.weight_decay, args.sche_step, args.sche_gamma)
    party_b_local_model = get_local_model(2, False, learning_rate, adj_matrix, 2, args.node_num, args.ReducedD, args.Q, args.weight_decay, args.sche_step, args.sche_gamma)
    party_a_global_model = MLP(learning_rate, args.Q, args.node_num, args.ReducedD, args.weight_decay, args.sche_step, args.sche_gamma).to(device)
    
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


def get_local_model(num, MULTIGRAPH, learning_rate, ADJPATH, o, node_num, reduced_d, Q, weight_decay, sche_step, sche_gamma):
    GRAPHNUMBER = 1 + int(MULTIGRAPH)
    ks, kt, bs, T, n, p = 3, 3, [[num, 16, 64], [64, 16, 64]], args.TIMESTEP_IN, node_num, 0
    Lk_new = torch.empty(GRAPHNUMBER,3,node_num,node_num).to(device)
    adjpathlist = []
    adjpathlist.append(ADJPATH)
    if MULTIGRAPH:
        adjpathlist.append('path to another adjacency matrix') # add another path here if using multiple graphs
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
    model = STGCN(ks, kt, bs, T, n, Lk_new, p, learning_rate, o, reduced_d, Q, weight_decay, sche_step, sche_gamma).to(device)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Q', type=int, default=1, help='the number of local updates for MP sub-models')
    parser.add_argument('--ReducedD', type=int, default=9, help='the reduced dimension of vfl model')
    parser.add_argument('--PRMP', type=float, default=0.2, help='penetration rate of MP')
    parser.add_argument('--LEARN', type=float, default=0.0003, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='the weight decay')
    parser.add_argument('--sche_step', type=int, default=140, help='the step size for learning rate scheduler')
    parser.add_argument('--sche_gamma', type=float, default=0.99, help='the gamma for learning rate scheduler')
    parser.add_argument('--TIMESTEP_OUT', type=int, default=1, help='the number of time steps to predict')
    parser.add_argument('--TIMESTEP_IN', type=int, default=9, help='the number of time steps to use as input')
    parser.add_argument('--BATCHSIZE', type=int, default=128, help='the batch size')
    parser.add_argument('--EPOCH', type=int, default=500, help='the number of epochs')
    parser.add_argument('--node_num', type=int, default=17, help='the number of nodes in the graph')
    parser.add_argument('--KEYWORD', type=str, default='fedtse', help='the keyword for saving the results')
    args = parser.parse_args()
    
    # set the seed
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    
    # set the device
    device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device("cpu")
    print("device: {0}".format(device))

    adj_matrix = './adjacency_matrix.csv'
    
    # set the keyword for saving the results
    KEYWORD = 'fedtse' + '_' + f'MP_{args.Q}' + f'_{args.LEARN}' + '_' + datetime.now().strftime("%y%m%d%H%M")
    PATH = './save/' + KEYWORD
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    
    # load data
    print("############################### Prepare Data #############################")
    train_data, val_data, test_data = load_data(args.PRMP)
    
    # train vfl model
    print("############################# Experiment start ###########################")
    run_experiment(
        train_data=train_data, 
        val_data=val_data, 
        test_data=test_data, 
        batch_size=args.BATCHSIZE, 
        learning_rate=args.LEARN, 
        epoch=args.EPOCH,
        adj_matrix=adj_matrix
        )