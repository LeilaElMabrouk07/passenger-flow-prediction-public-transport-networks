import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    def __init__(self, c_in, c_out, kt, dropout):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(kt, 1), padding=(kt//2, 0))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.dropout(self.act(self.conv(x)))

class GraphConv(nn.Module):
    def __init__(self, A_norm, c_in, c_out):
        super().__init__()
        self.A = A_norm  
        self.lin = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x):
        x = torch.einsum("nm,bctm->bctn", self.A, x)
        return self.lin(x)

class STGCN(nn.Module):
    def __init__(self, num_nodes, A_norm, in_channels=1, hidden_channels=32, kt=3, dropout=0.1, horizon=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        self.t1 = TemporalConv(in_channels, hidden_channels, kt, dropout)
        self.g1 = GraphConv(A_norm, hidden_channels, hidden_channels)
        self.t2 = TemporalConv(hidden_channels, hidden_channels, kt, dropout)

        self.head = nn.Conv2d(hidden_channels, horizon, kernel_size=(1, 1))


    def forward(self, x):
        x = x.permute(0, 1, 3, 2)

        h = self.t1(x)
        h = self.g1(h)
        h = torch.relu(h)
        h = self.t2(h)

        h_last = h[:, :, -1:, :]
        y = self.head(h_last)       
        y = y.squeeze(2)          
        return y

