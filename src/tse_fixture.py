import numpy as np
from .utils import save_loss, plot_loss, plot_pred, inverse_normalize


class TSEFixture(object):

    def __init__(self, learning):
        self.learning = learning

    def fit(self, train_data, val_data, test_data, epochs=50, batch_size=-1, PATH=None, scaler=None):

        main_party_id = self.learning.get_main_party_id()
        Xa_train = train_data[main_party_id]["X"]
        y_train = train_data[main_party_id]["Y"]
        Xa_val = val_data[main_party_id]["X"]
        y_val = val_data[main_party_id]["Y"]
        Xa_test = test_data[main_party_id]["X"]
        y_test = test_data[main_party_id]["Y"]
        
        y_val = inverse_normalize(scaler, y_val)
        y_test = inverse_normalize(scaler, y_test)
        y_train_cpu = y_train.clone().cpu().detach().numpy()
        # y_train_cpu_on_cpu = y_train_cpu.cpu()
        # y_train_cpu_detached = y_train_cpu_on_cpu.detach()
        # y_train_cpu_np = y_train_cpu_detached.numpy()
        y_train_cpu = inverse_normalize(scaler, y_train_cpu)


        # debug
        Xa_test_15 = Xa_test.clone()
        Xa_test_15[:,0:1,:,:] = Xa_test_15[:,0:1,:,:]
        Xa_test_15[:,1:3,:,:] = Xa_test_15[:,1:3,:,:]*1.2

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
                # print("Xa_batch shape:", Xa_batch.shape)
                # compute training loss
                train_loss = self.learning.fit(Xa_batch, Y_batch,
                                                   global_step)
                train_loss_list.append(train_loss)

            # compute training loss
            ave_train_rmse = np.sqrt(np.mean(train_loss_list))
        
            # compute validation loss
            val_y_preds = self.learning.predict(Xa_val)
            # compute the mse of the validation set
            # but val_y_preds and y_val are tensors, so we need to convert them to numpy array
            val_y_preds = val_y_preds.cpu().detach().numpy()
            val_y_preds = inverse_normalize(scaler, val_y_preds)

            val_rmse_q = np.sqrt(np.mean(np.square(val_y_preds[:, 0, :, :]
                                                   - y_val[:, 0, :, :])))
            val_rmse_k = np.sqrt(np.mean(np.square(val_y_preds[:, 1, :, :]
                                                   - y_val[:, 1, :, :])))

            ####### debug #######
            test_y_preds_15 = self.learning.predict(Xa_test_15)
            test_y_preds_15 = test_y_preds_15.cpu().detach().numpy()
            test_y_preds_15 = inverse_normalize(scaler, test_y_preds_15)
            ####### debug #######

            # compute test loss
            test_y_preds = self.learning.predict(Xa_test)
            # compute the mse of the test set
            # but test_y_preds and y_test are tensors, so we need to convert them to numpy array
            test_y_preds = test_y_preds.cpu().detach().numpy()
            test_y_preds = inverse_normalize(scaler, test_y_preds)
            test_rmse_q = np.sqrt(np.mean(np.square(test_y_preds[:, 0, :, :] - y_test[:, 0, :, :])))
            test_rmse_k = np.sqrt(np.mean(np.square(test_y_preds[:, 1, :, :] - y_test[:, 1, :, :])))
            # MAPE
            test_mape_q = np.mean(np.abs((test_y_preds[:, 0, :, :] - y_test[:, 0, :, :]) / y_test[:, 0, :, :])) * 100
            test_mape_k = np.mean(np.abs((test_y_preds[:, 1, :, :] - y_test[:, 1, :, :]) / y_test[:, 1, :, :])) * 100
            # test_rmse = np.sqrt(np.mean(np.square(test_y_preds - y_test)))
            # test_y_preds.shape (batch, 2, 1, 186)
                
            print("===> epoch: {0}, train_loss: {1}, val_loss_q: {2}, val_loss_k: {3}, test_loss_q: {4}, test_loss_k: {5}, test_mape_q: {6}, test_mape_k: {7}".format(ep, ave_train_rmse, 
                                                                                                                                                                      val_rmse_q, val_rmse_k, 
                                                                                                                                                                      test_rmse_q, test_rmse_k,
                                                                                                                                                                      test_mape_q, test_mape_k))
            # compute train loss
            train_y_preds = self.learning.predict(Xa_train)
            train_y_preds = train_y_preds.cpu().detach().numpy()
            train_y_preds = inverse_normalize(scaler, train_y_preds)
            train_rmse_q = np.sqrt(np.mean(np.square(train_y_preds[:, 0, :, :] - y_train_cpu[:, 0, :, :])))
            train_rmse_k = np.sqrt(np.mean(np.square(train_y_preds[:, 1, :, :] - y_train_cpu[:, 1, :, :])))
            
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
                f.write("===> epoch: {0}, train_loss: {1}, val_loss_q: {2}, val_loss_k: {3}, test_loss_q: {4}, test_loss_k: {5}, test_mape_q: {6}, test_mape_k: {7}".format(ep, ave_train_rmse, 
                                                                                                                                                                            val_rmse_q, val_rmse_k, 
                                                                                                                                                                            test_rmse_q, test_rmse_k,                                                                                                                                     test_mape_q, test_mape_k))
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


        # np.save(PATH + '/' + 'test_y_preds_15.npy', test_y_preds_15)

        ####### debug #######
        # save the train pred
        # Xa_train_15 = Xa_train
        # Xa_train_15[:,0:1,:,:] = Xa_train_15[:,0:1,:,:]
        # Xa_train_15[:,1:3,:,:] = Xa_train_15[:,1:3,:,:]*1.5
        # train_y_preds_15 = self.learning.predict(Xa_train_15)
        # train_y_preds_15 = train_y_preds_15.cpu().detach().numpy()
        # train_y_preds_15 = inverse_normalize(scaler, train_y_preds_15)
        # np.save(PATH + '/' + 'train_y_preds_15.npy', train_y_preds_15)
        ####### debug #######

