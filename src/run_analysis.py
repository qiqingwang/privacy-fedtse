import os
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
time series analysis
MP 10% - 100%
queuing diagram
density 几辆车
acf
multi-task learning?
'''

def find_your_path():
    # obtain the absolute path of the script
    script_path = os.path.abspath(__file__)
    # obtain the directory of the script
    script_dir = os.path.dirname(script_path)
    # change the current working directory to the script directory
    os.chdir(script_dir)


def plot_pred_one(in_path, out_path, idx):
    
    plt.style.use('seaborn-bright')
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 4
    truth = np.load(os.path.join(in_path, 'test_y_label.npy'))
    pred = np.load(os.path.join(in_path, 'test_y_preds.npy'))

    Q_truth = truth[:, 0, :, idx]
    K_truth = truth[:, 1, :, idx]
    Q_pred = pred[:, 0, :, idx]
    K_pred = pred[:, 1, :, idx]
    Q_truth_mean = np.squeeze(Q_truth)
    K_truth_mean = np.squeeze(K_truth)
    Q_pred_mean = np.squeeze(Q_pred)
    K_pred_mean = np.squeeze(K_pred)

    # mse
    mse = np.sqrt(np.mean(np.square(K_truth_mean - K_pred_mean)))
    print('mse: ', mse)
 # test_loss_q: 12.748712852211765, test_loss_k: 27.23348117264934
    # transform Q_truth_mean to dataframe
    # Q_truth_mean = pd.DataFrame(Q_truth_mean)
    # Q_pred_mean = pd.DataFrame(Q_pred_mean)
    # save to csv
    # Q_truth_mean.to_csv(os.path.join(out_path, 'Q_truth_mean.csv'))
    # Q_pred_mean.to_csv(os.path.join(out_path, 'Q_pred_mean.csv'))
    

    # print(Q_pred_mean[])

    # plot Q
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.plot(Q_truth_mean[:], label='Ground-truth flow', linewidth=3)
    ax.plot(Q_pred_mean[:], label='Predicted flow', linewidth=3)
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set(xlabel='Timesteps', ylabel='Flow')
    # ax.set_ylim([0, 10000])
    ax.legend(loc='best', fontsize=20)
    # fig.savefig(
    #     os.path.join(out_path, 'pneuma-pred-q.png'),
    #     bbox_inches='tight',
    # )
    plt.show()

    # plot K
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.plot(K_truth_mean[:], label='Ground-truth density', linewidth=3)
    ax.plot(K_pred_mean[:], label='Predicted density', linewidth=3)
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set(xlabel='Timesteps', ylabel='Density')
    # ax.set_ylim([0, 500])
    ax.legend(loc='best', fontsize=20)
    # fig.savefig(
    #     os.path.join(out_path, 'pneuma-pred-k.png'),
    #     bbox_inches='tight',
    # )
    plt.show()


if __name__ == '__main__':
    find_your_path()
    # save/FedTSE_pneuma_MA_1_MP_0.2_0.001_1_2311031851
    path1 = '../save/FedTSE_pneuma_MA_1_MP_1.0_0.001_1_2311061919'
    plot_pred_one(path1, path1, 1)

