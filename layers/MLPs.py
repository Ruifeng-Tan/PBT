import torch
from torch import nn
class FlattenLinear(nn.Module):
    def __init__(self, gate_d_ff, input_dim, output_dim):
        super(FlattenLinear, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.DKP_linear = nn.Sequential(nn.Linear(gate_d_ff, output_dim), nn.Sigmoid())
    
    def forward(self, x, DKP_embeddings):
        out = self.linear(x) * self.DKP_linear(DKP_embeddings).unsqueeze(1)
        return out

class BatteryMoEMLP(nn.Module):
    def __init__(self, gate_d_ff, in_dim, hidden_dim, drop_rate, activation='gelu'):
        super(BatteryMoEMLP, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.act_linear = nn.Linear(in_dim, hidden_dim)
        self.DKP_linear = nn.Sequential(nn.Linear(gate_d_ff, hidden_dim), nn.Sigmoid())
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x, DKP_embeddings):
        '''
        x: [B, *, in_dim]
        '''
        out = self.act_linear(x) * self.DKP_linear(DKP_embeddings).unsqueeze(1)
        out = self.dropout(out)
        out = self.out_linear(out)
        return out

class MLPBlockGELU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation='gelu'):
        super(MLPBlockGELU, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.act_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'swish' or activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.out_linear = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.act(self.act_linear(x))
        out = self.dropout(out)
        out = self.out_linear(out)
        return out
    
class MLPBlockSwishGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlockSwishGLU, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.act = nn.Sigmoid()
        self.act_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, in_dim, bias=False)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        act_x = self.in_linear(x)

        # SwishGLU
        out = act_x * self.act(act_x) * self.act_linear(x)

        out = self.dropout(out)
        out = self.out_linear(out)
        return out