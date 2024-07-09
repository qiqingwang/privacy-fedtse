import numpy as np
import pandas as pd
from .utils import save_loss, plot_loss, plot_pred, inverse_normalize

class FederatedLearningFixture(object):

    def __init__(self, federated_learning):
        self.federated_learning = federated_learning

    def fit(self, train_data, val_data, test_data, epochs=50, batch_size=-1, PATH=None, scaler=None):

        main_party_id = self.federated_learning.get_main_party_id()
        Xa_train = train_data[main_party_id]["X"]
        y_train = train_data[main_party_id]["Y"]
        Xa_val = val_data[main_party_id]["X"]
        y_val = val_data[main_party_id]["Y"]
        Xa_test = test_data[main_party_id]["X"]
        y_test = test_data[main_party_id]["Y"]

        y_val = inverse_normalize(scaler, y_val)
        y_test = inverse_normalize(scaler, y_test)
        y_train_cpu = y_train.clone().cpu().detach().numpy()
        y_train_cpu = inverse_normalize(scaler, y_train_cpu)

        N = Xa_train.shape[0]
        residual = N % batch_size
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        print("number of samples:", N)
        print("batch size:", batch_size)
        print("number of batches:", n_batches)

        global_step = -1

        epochepoch = []
        train_losses = []
        val_losses_q = []
        val_losses_k = []
        test_losses_q = []
        test_losses_k = []
        train_losses_q = []
        train_losses_k = []

        for ep in range(epochs):
            train_loss_list = []

            for batch_idx in range(n_batches):
                global_step += 1

                # prepare batch data for party A, which has both X and y.
                Xa_batch = Xa_train[batch_idx * batch_size: batch_idx * batch_size + batch_size]
                Y_batch = y_train[batch_idx * batch_size: batch_idx * batch_size + batch_size]

                # prepare batch data for all other parties, which only has both X.
                party_X_train_batch_dict = dict()
                for party_id, party_X in train_data["party_list"].items():
                    party_X_train_batch_dict[party_id] = party_X[batch_idx * batch_size: batch_idx * batch_size + batch_size]
                # compute training loss
                train_loss = self.federated_learning.fit(Xa_batch, Y_batch, party_X_train_batch_dict, global_step)
                train_loss_list.append(train_loss)
                
            ### compute training loss
            ave_train_rmse = np.sqrt(np.mean(train_loss_list))
        
            ### compute validation loss
            party_X_val_dict = dict()
            for party_id, party_X in val_data["party_list"].items():
                party_X_val_dict[party_id] = party_X
            val_y_preds = self.federated_learning.predict(Xa_val, party_X_val_dict)
            # compute the mse of the validation set
            val_y_preds = val_y_preds.cpu().detach().numpy()
            val_y_preds = inverse_normalize(scaler, val_y_preds)
            val_rmse_q = np.sqrt(np.mean(np.square(val_y_preds[:,0,:,:] - y_val[:,0,:,:])))
            val_rmse_k = np.sqrt(np.mean(np.square(val_y_preds[:,1,:,:] - y_val[:,1,:,:])))

            ### compute test loss
            party_X_test_dict = dict()
            for party_id, party_X in test_data["party_list"].items():
                party_X_test_dict[party_id] = party_X
            test_y_preds = self.federated_learning.predict(Xa_test, party_X_test_dict)
            # compute the mse of the test set
            test_y_preds = test_y_preds.cpu().detach().numpy()
            test_y_preds = inverse_normalize(scaler, test_y_preds)
            test_rmse_q = np.sqrt(np.mean(np.square(test_y_preds[:,0,:,:] - y_test[:,0,:,:])))
            test_rmse_k = np.sqrt(np.mean(np.square(test_y_preds[:,1,:,:] - y_test[:,1,:,:])))

            print("===> epoch: {0}, train_loss: {1}, val_loss_q: {2}, val_loss_k: {3}, test_loss_q: {4}, test_loss_k: {5}".format(ep, ave_train_rmse, 
                                                                                                                                  val_rmse_q, val_rmse_k, 
                                                                                                                                  test_rmse_q, test_rmse_k))
            # compute train loss
            party_X_test_dict_1 = dict()
            for party_id, party_X in train_data["party_list"].items():
                party_X_test_dict_1[party_id] = party_X
            train_y_preds = self.federated_learning.predict(Xa_train, party_X_test_dict_1)
            train_y_preds = train_y_preds.cpu().detach().numpy()
            train_y_preds = inverse_normalize(scaler, train_y_preds)
            train_rmse_q = np.sqrt(np.mean(np.square(train_y_preds[:,0,:,:] - y_train_cpu[:,0,:,:])))
            train_rmse_k = np.sqrt(np.mean(np.square(train_y_preds[:,1,:,:] - y_train_cpu[:,1,:,:])))

            epochepoch.append(ep)
            train_losses.append(ave_train_rmse)
            val_losses_q.append(val_rmse_q)
            val_losses_k.append(val_rmse_k)
            test_losses_q.append(test_rmse_q)
            test_losses_k.append(test_rmse_k)
            train_losses_q.append(train_rmse_q)
            train_losses_k.append(train_rmse_k)

            # save test prediction and test label
            with open(PATH + '/' + '_log.txt', 'a') as f:
                f.write("===> epoch: {0}, train_loss: {1}, val_loss_q: {2}, val_loss_k: {3}, test_loss_q: {4}, test_loss_k: {5}".format(ep, ave_train_rmse, 
                                                                                                                                        val_rmse_q, val_rmse_k, 
                                                                                                                                        test_rmse_q, test_rmse_k
                                                                                                                                        ))
                f.write('\n')
        
        # save the loss and prediction
        save_loss(epochepoch, val_losses_q, val_losses_k, test_losses_q, test_losses_k, 
                  train_losses_q, train_losses_k, PATH)
        np.save(PATH + '/' + 'test_y_preds.npy', test_y_preds)
        np.save(PATH + '/' + 'test_y_label.npy', y_test)
        np.save(PATH + '/' + 'train_y_preds.npy', train_y_preds)
        np.save(PATH + '/' + 'train_y_label.npy', y_train_cpu)
        # plot the loss and prediction
        plot_pred(PATH, PATH, mode='test')
        plot_pred(PATH, PATH, mode='train')
        plot_loss(PATH, PATH)
