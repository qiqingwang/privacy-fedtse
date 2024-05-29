import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        elif self.act == "sigmoid":
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid": 
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # print('spatio_conv_layer Lk.shape' + str(self.Lk.shape))
        # print('spatio_conv_layer x.shape' + str(x.shape))
        
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        # print('x_c shape is', x_c.shape)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        
        # print('sp shape is :', x_gc.shape)
        return torch.relu(x_gc + x)


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class st_conv_block(nn.Module):
    # self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconvlist = nn.ModuleList()
        self.sconvseq = nn.Sequential()
        for i in range(len(Lk)):
            self.sconvseq.add_module('spatio_conv_layer', spatio_conv_layer(ks, c[1], Lk[i]))
        # print(len(Lk))
        # print('c is', c)
        for i in range(len(Lk)):
            self.sconvlist.append(spatio_conv_layer(ks, c[1], Lk[i]))
        self.sconvtest = spatio_conv_layer(ks, c[1], Lk[0])
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.mlp = linear(c[1]*len(Lk), c[1])
        # def __init__(self, kt, c_in, c_out, act="relu"):
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)
        self.Lk = Lk

    def forward(self, x):
        x_t1 = self.tconv1(x)
        # print('x_t1 shape is',x_t1.shape)
        # x_s = self.sconvlist(x_t1)
        # x_s = self.sconvseq(x_t1)
        # x_s = torch.empty(len(self.Lk), x_t1.shape).to(device)
        # print('x_s shape before', x_s.shape)
        # x_s = torch.zeros_like(x_t1)
        x_s = torch.empty_like(x_t1)
        for i,layer in enumerate(self.sconvlist):
            # x_s += layer(x_t1)
            # print(i)
            if i == 0:
                x_s = layer(x_t1)
            else:
                x_s = torch.cat((x_s, layer(x_t1)), 1)
        # print('x_s shape before mlp', x_s.shape)
        
        x_s = self.mlp(x_s)
        # print('x_s shape after mlp', x_s.shape)
        # x_s = x_t1
        # x_s = self.sconvtest(x_t1)
        # x_s = self.sconv(x_t1)
        # print('st_conv_block x_s shape is',x_s.shape)
        x_t2 = self.tconv2(x_s)
        # print('x_t2 shape is', x_t2.shape)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        # print('x_ln shape is', x_ln.shape)
        # print('x_ln shape is xxxxxxxxxxxxx', self.dropout(x_ln).shape)       
        return self.dropout(x_ln)


class output_layer(nn.Module):
    def __init__(self, c, T, n, o, mode='vfl'):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        # 影响模型的输出维度
        self.fc = nn.Conv2d(c, o, (1, 1))
        self.relu = nn.ReLU()
        self.mode = mode

        if mode == 'vfl':
            self.mlp1 = nn.Linear(2*17, 1*9)
        else:
            self.mlp1 = nn.Linear(2*17, 2*17)

        # self.fc1 = nn.Conv2d(64, 16, (1, 1))
        # self.fc2 = nn.Conv2d(16, o, (1, 1))
        # self.pooling = nn.MaxPool2d((1, 2), 1, return_indices=True)
        # self.unpooling = nn.MaxUnpool2d((1, 2), 1)
        

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        x = self.fc(x_t2)

        x = x.reshape(-1, 2*17)
        x = self.mlp1(x)
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    # model = STGCN(ks, kt, bs, T, n, Lk, p).to(device)
    # bs = [[CHANNEL, 16, 64], [64, 16, 64]]
    def __init__(self, ks, kt, bs, T, n, Lk, p, lr, o, mode, Q):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n, o, mode)
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1400, gamma=0.99)
        self.Q = Q

    def forward(self, x):
        # print('x.shape is', x.shape)
        x_st1 = self.st_conv1(x)
        # print('x_st1.shape is ', x_st1.shape
        x_st2 = self.st_conv2(x_st1)
        # print('x_st2.shape is ', x_st2.shape)
        output_data = self.output(x_st2)
        # print('output.shape is', output_data.shape)
        return output_data
    
    def predict(self, x):
        return self.forward(x)
    
    def backward(self, x, grads):
        # the zero_grad() here is important, otherwise the grad will accumulate as the global_model is backwarded once already
        self.optimizer.zero_grad()
        output = self.forward(x) # time consuming ?
        output.backward(grads)

        ############################ check for gards
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        #     else:
        #         print(name, "None")
        for i in range(self.Q):
            self.optimizer.step()
        self.scheduler.step()
        
        x_grad = x.grad
        return x_grad


def weight_matrix(W, sigma2=0.1, epsilon=0.5, alpha=10):
    '''
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    n = W.shape[0]
    W = W /alpha
    W[W==0]=np.inf
    W2 = W * W
    W_mask = (np.ones([n, n]) - np.identity(n))
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)
