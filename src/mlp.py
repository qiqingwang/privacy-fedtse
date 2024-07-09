import torch
import torch.nn as nn

# check the parameters settings in STGCN !
class MLP(nn.Module):
    def __init__(self, LEARN, Q, node_num, reduced_d, weight_decay, sche_step, sche_gamma):
        super(MLP, self).__init__()
        self.reduced_d = reduced_d
        self.node_num = node_num
        self.fc1 = nn.Linear(2*reduced_d, 16*node_num)
        self.fc2 = nn.Linear(16*node_num, 2*node_num)
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARN, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sche_step, gamma=sche_gamma)
        self.Q = Q
    
    def forward(self, x):
        x = x.reshape(-1, 2*self.reduced_d)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(-1, 2, 1, self.node_num)
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