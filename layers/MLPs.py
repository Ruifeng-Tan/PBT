import torch
from torch import nn

class BatteryMoEMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU() if hidden_act == 'gelu' else nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

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