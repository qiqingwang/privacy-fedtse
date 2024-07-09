import torch
import torch.nn as nn


class VFLHostMA(object):
    def __init__(self, local_model, global_model):
        super(VFLHostMA, self).__init__()
        self.localModel = local_model
        self.is_debug = False
        self.criterion = nn.MSELoss()
        self.global_model = global_model
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
        # MA appends its own component
        self.parties_grad_component_list.append(self.Z_0)
        self.dim_grad = self.Z_0.shape[1] # the dimention will be used to split the global grad to local grads

        # MA forward parties_grad_component_list to a global model
        self.Z_all = torch.cat(self.parties_grad_component_list, dim=1)
        self.Z_all.retain_grad()
        self.Z_fused = self.global_model.forward(self.Z_all)

        self._compute_common_gradient_and_loss(y)
        self._update_models(X, y)

    def predict(self, X, component_list):
        Z_0_pred = self.localModel.forward(X)
        component_list.append(Z_0_pred)
        Z_all_pred = torch.cat(component_list, dim=1)
        Z_fused_pred = self.global_model.forward(Z_all_pred)
        return Z_fused_pred

    def receive_components(self, component_list):
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component)

    def fit(self):
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []

    def _compute_common_gradient_and_loss(self, y):
        U = self.Z_fused
        mse_loss = self.criterion(U, y)
        grads = torch.autograd.grad(outputs=mse_loss, inputs=U)
        self.top_grads = grads[0]
        self.loss = mse_loss.item()

    def send_gradients(self):
        return self.back_grad_wrt_Z[:, :self.dim_grad]

    def _update_models(self, X, y):
        self.back_grad_wrt_Z = self.global_model.backward(self.Z_all, self.top_grads)
        # cut the grad of MA's own component
        back_grad_wrt_Z_0 = self.back_grad_wrt_Z[:, -self.dim_grad:]
        self.localModel.backward(X, back_grad_wrt_Z_0)

    def get_loss(self):
        return self.loss


class VFLGuestMP(object):
    def __init__(self, local_model):
        super(VFLGuestMP, self).__init__()
        self.localModel = local_model
        self.is_debug = False
        self.common_grad = None
        self.current_global_step = None
        self.X = None

    def set_batch(self, X, global_step):
        self.X = X
        self.current_global_step = global_step

    def _forward_computation(self, X):
        Z_k = self.localModel.forward(X)
        return Z_k

    def _fit(self, X, y):
        self.localModel.backward(X, self.common_grad)

    def receive_gradients(self, gradients):
        self.common_grad = gradients
        self._fit(self.X, None)

    def send_components(self):
        return self._forward_computation(self.X)

    def predict(self, X):
        return self._forward_computation(X)


class FedTSE(object):
    def __init__(self, party_A, main_party_id="_main"):
        super(FedTSE, self).__init__()
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

    def add_party(self, *, id, party_model):
        self.party_dict[id] = party_model

    def fit(self, X_A, y, party_X_dict, global_step):
        if self.is_debug: print("==> start fit")

        # set batch data for party A.
        # only party A has label.
        self.party_a.set_batch(X_A, y, global_step)

        # set batch data for all other parties
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, global_step)

        if self.is_debug: print("==> Host receive intermediate computing results from guests")
        comp_list = []
        for party in self.party_dict.values():
            comp_list.append(party.send_components())
        self.party_a.receive_components(component_list=comp_list)

        if self.is_debug: print("==> Host train and computes loss")
        self.party_a.fit()
        loss = self.party_a.get_loss()

        if self.is_debug: print("==> Host sends out common grad")
        grad_result = self.party_a.send_gradients()

        if self.is_debug: print("==> Guests receive common grad from guest and perform training")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result)

        return loss
    
    def predict(self, X_A, party_X_dict):
        comp_list = []
        for id, party_X in party_X_dict.items():
            comp_list.append(self.party_dict[id].predict(party_X))
        return self.party_a.predict(X_A, component_list=comp_list)