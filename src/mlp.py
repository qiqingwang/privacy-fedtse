import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(FusionModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # batch_size, height, width = x.shape
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        out = x.reshape(batch_size, 2, height, width)
        # out = x.reshape(batch_size, 2, 1, width)
        return out
    
    def backward(self, x, grads):
        # the zero_grad() here is important, otherwise the grad will accumulate as the global_model is backwarded once already
        self.optimizer.zero_grad()
        output = self.forward(x) # time consuming ?
        output.backward(grads)
        self.l1_regularization()
        # self.l2_regularization()
        self.optimizer.step()
        x_grad = x.grad
        return x_grad
    

    # def l1_regularization(self):
    #     for param in self.parameters():
    #         param.grad.data.add_(0.001 * torch.sign(param.data))

    def l1_regularization(self):
        for name, param in self.named_parameters():
            if name[-5:] == 'eight':
                param.grad.data.add_(0.00001 * torch.sign(param.data))

    def l2_regularization(self):
        for param in self.parameters():
            param.grad.data.add_(0.001 * param.data)
    

class FusionModel_test(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(FusionModel_test, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        out = x.reshape(batch_size, 2, width)
        return out
    
    def backward(self, x, grads):
        # the zero_grad() here is important, otherwise the grad will accumulate as the global_model is backwarded once already
        self.optimizer.zero_grad()
        output = self.forward(x) # time consuming ?
        output.backward(grads)
        self.optimizer.step()
        x_grad = x.grad
        return x_grad