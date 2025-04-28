import torch
from torch import nn

class BatteryMoESwishGLU(nn.Module):
    def __init__(self, in_dim, d_llm, hidden_dim, drop_rate, beta=1.0):
        super(BatteryMoESwishGLU, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.DKP_linear = nn.Linear(d_llm, hidden_dim, bias=True)
        self.beta = beta

        self.V_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.act = nn.Sigmoid()
        self.out_linear = nn.Linear(hidden_dim, in_dim, bias=False)
    
    def forward(self, x, DKP_embeddings):
        '''
        x: [B, *, in_dim]
        DKP_embeddings: [B, d_llm]
        '''
        act_x = self.in_linear(x)
        act_DKP = self.DKP_linear(DKP_embeddings)
        if act_x.dim() == 3:
            act_DKP = act_DKP.unsqueeze(1)
        # SwishGLU
        out = act_x * self.act(act_DKP*self.beta) * self.V_linear(x)

        out = self.dropout(out)
        out = self.out_linear(out)
        return out

class MLPBlockGELU(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate, activation):
        super(MLPBlockGELU, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.act_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
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