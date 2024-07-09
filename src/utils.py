import os
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

def save_loss(epochepoch, val_losses_q, val_losses_k, test_losses_q, test_losses_k, train_losses_q, train_losses_k, PATH):
    data = {
            'Epoch': epochepoch,
            'Validation Loss Q': val_losses_q,
            'Validation Loss K': val_losses_k,
            'Test Loss Q': test_losses_q,
            'Test Loss K': test_losses_k,
            'Train Loss Q': train_losses_q,
            'Train Loss K': train_losses_k
        }
    df = pd.DataFrame(data)
    df.to_csv(PATH+'/'+'loss.csv', index=False)


def plot_loss(in_path, out_path):
    df = pd.read_csv(os.path.join(in_path, 'loss.csv'))
    epochepoch = df['Epoch'].values
    val_losses_q = df['Validation Loss Q'].values
    val_losses_k = df['Validation Loss K'].values
    test_losses_q = df['Test Loss Q'].values
    test_losses_k = df['Test Loss K'].values
    train_losses_q = df['Train Loss Q'].values
    train_losses_k = df['Train Loss K'].values

    # plt.style.use('seaborn-bright')
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 4
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[32, 10])
    # q
    ax1.plot(epochepoch, val_losses_q, label='Validation loss (flow)', linewidth=3)
    ax1.plot(epochepoch, test_losses_q, label='Test loss (flow)', linewidth=3)
    ax1.plot(epochepoch, train_losses_q, label='Train loss (flow)', linewidth=3)
    ax1.tick_params(axis='x', labelrotation=0)
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.legend(loc='best', fontsize=20)
    ax1.set_title('Losses for q')
    # k
    ax2.plot(epochepoch, val_losses_k, label='Validation loss (density)', linewidth=3)
    ax2.plot(epochepoch, test_losses_k, label='Test loss (density)', linewidth=3)
    ax2.plot(epochepoch, train_losses_k, label='Train loss (density)', linewidth=3)
    ax2.tick_params(axis='x', labelrotation=0)
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(loc='best', fontsize=20)
    ax2.set_title('Losses for k')

    fig.savefig(
        os.path.join(out_path, 'loss.png'),
        bbox_inches='tight',
    )
    fig.show()


def plot_pred(in_path, out_path, mode='train'):
    # plt.style.use('seaborn-bright')
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 4
    if mode == 'train':
        truth = np.load(os.path.join(in_path, 'train_y_label.npy'))
        pred = np.load(os.path.join(in_path, 'train_y_preds.npy'))
        fig_name_q = 'pred-q-train.png'
        fig_name_k = 'pred-k-train.png'
    elif mode == 'test':
        truth = np.load(os.path.join(in_path, 'test_y_label.npy'))
        pred = np.load(os.path.join(in_path, 'test_y_preds.npy'))
        fig_name_q = 'pred-q-test.png'
        fig_name_k = 'pred-k-test.png'

    Q_truth = truth[:, 0, :, :]
    K_truth = truth[:, 1, :, :]
    Q_pred = pred[:, 0, :, :]
    K_pred = pred[:, 1, :, :]
    Q_truth = np.squeeze(Q_truth)
    K_truth = np.squeeze(K_truth)
    Q_pred = np.squeeze(Q_pred)
    K_pred = np.squeeze(K_pred)
    Q_truth_mean = np.mean(Q_truth, axis=1)
    K_truth_mean = np.mean(K_truth, axis=1)
    Q_pred_mean = np.mean(Q_pred, axis=1)
    K_pred_mean = np.mean(K_pred, axis=1)

    # plot Q
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.plot(Q_truth_mean[:], label='Ground-truth flow', linewidth=3)
    ax.plot(Q_pred_mean[:], label='Predicted flow', linewidth=3)
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set(xlabel='Timesteps', ylabel='Flow')
    ax.legend(loc='best', fontsize=20)
    fig.savefig(
        os.path.join(out_path, fig_name_q),
        bbox_inches='tight',
    )
    fig.show()

    # plot K
    fig, ax = plt.subplots(figsize=[16, 10])
    ax.plot(K_truth_mean[:], label='Ground-truth density', linewidth=3)
    ax.plot(K_pred_mean[:], label='Predicted density', linewidth=3)
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set(xlabel='Timesteps', ylabel='Density')
    ax.legend(loc='best', fontsize=20)
    fig.savefig(
        os.path.join(out_path, fig_name_k),
        bbox_inches='tight',
    )
    fig.show()


def inverse_normalize(scaler, y):
    # scaler = scaler_4.mean_, scaler_4.scale_, scaler_5.mean_, scaler_5.scale_
    y[:,0,:,:] = y[:,0,:,:] * scaler[1][0] + scaler[0][0]
    y[:,1,:,:] = y[:,1,:,:] * scaler[3][0] + scaler[2][0]
    return y


