import torch
from torch import nn

class MLPBlockSwishGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlockSwishGLU, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.act_linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.Sigmoid()
        self.act_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)

        # SwishGLU
        out = out * self.act(self.act_linear1(out)) * self.act_linear2(out)

        out = self.dropout(out)
        out = self.out_linear(out)
        return out