# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, lr):
#         super(GRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.lr = lr
#         self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
#         self.fc = nn.Linear(self.hidden_size, self.output_size)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

#     def forward(self, x):
#         batch, _, _, _ = x.shape
#         x = x.reshape(batch, -1, self.input_size)
#         x, _ = self.gru(x)
#         x = self.fc(x)
#         x = x.reshape(batch, 2, 1, 3)
#         return x
    
#     def backward(self, x, grads):
#         self.optimizer.zero_grad()
#         output = self.forward(x)
#         output.backward(grads)
#         self.optimizer.step()