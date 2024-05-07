import torch
import torch.nn as nn


class HostMA(object):

    def __init__(self, local_model, global_model):
        super(HostMA, self).__init__()
        self.localModel = local_model
        self.global_model = global_model
        self.is_debug = False
        self.criterion = nn.MSELoss()
        self.parties_grad_component_list = []
        self.current_global_step = None
        self.X = None
        self.y = None
        self.dim_grad = None

    def set_batch(self, X, y, global_step):
        self.X = X
        self.y = y
        self.current_global_step = global_step

    def _fit(self, X, y):
        self.Z_0 = self.localModel.forward(X)
        # print("Z_0.shape: ", self.Z_0.shape)
        # MA appends its own component
        # self.Z_all.retain_grad()
        self.Z_0.retain_grad()
        self.Z_fused = self.global_model.forward(self.Z_0)
        
        self._compute_common_gradient_and_loss(y)
        self._update_models(X, y)

    def predict(self, X):
        Z_0_pred = self.localModel.forward(X)
        Z_fused_pred = self.global_model.forward(Z_0_pred)
        return Z_fused_pred
        # return Z_0_pred

    def _compute_common_gradient_and_loss(self, y):
        U = self.Z_fused
        # U = self.Z_0
        
        
        mse_loss = self.criterion(U, y)
        
        # mse_loss = self.criterion(U[:,0,:,:], y[:,0,:,:]) + 20 * self.criterion(U[:,1,:,:], y[:,1,:,:])
        grads = torch.autograd.grad(outputs=mse_loss, inputs=U)
        self.top_grads = grads[0]
        self.loss = mse_loss.item()

    # def _L1_loss(self):
    #     # L1 loss
    #     regularization_loss = 0
    #     for param in self.global_model.parameters():
    #         regularization_loss += torch.sum(abs(param))
    #     print("regularization_loss: ", regularization_loss)
    #     return regularization_loss

    def _update_models(self, X, y):
        # Print out the gradients of each parameter in the network
        ###################################### check for grads
        # for name, param in self.localModel.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        #     else:
        #         print(name, "None")
        
        self.back_grad_wrt_Z = self.global_model.backward(self.Z_0, self.top_grads)

        ##################################### check for grads
        # for name, param in self.global_model.named_parameters():
        #     if param.grad is not None:
        #         if name == "fc3.weight" or name == "fc3.bias":
        #             print(name, param.grad)
            
        # cut the grad of MA's own component
        back_grad_wrt_Z_0 = self.back_grad_wrt_Z
        self.localModel.backward(X, back_grad_wrt_Z_0)

        # self.localModel.backward(X, self.top_grads)

    def get_loss(self):
        return self.loss

    def fit(self):
        self._fit(self.X, self.y)


class TSE(object):

    def __init__(self, party_A, main_party_id="_main"):
        super(TSE, self).__init__()
        self.main_party_id = main_party_id
        # party A is the parity with labels
        self.party_a = party_A
        # the party dictionary stores other parties that have no labels
        self.party_dict = dict()
        self.is_debug = False

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def get_main_party_id(self):
        return self.main_party_id

    def fit(self, X_A, y, global_step):
        if self.is_debug: print("==> start fit")

        # set batch data for party A.
        # only party A has label.
        self.party_a.set_batch(X_A, y, global_step)

        if self.is_debug: print("==> Host train and computes loss")
        self.party_a.fit()
        loss = self.party_a.get_loss()

        return loss
    
    ########################################################!!!!!!!!! start from this
    def predict(self, X_A):
        return self.party_a.predict(X_A)