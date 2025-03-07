import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super(GatingFusion, self).__init__()
        self.proj1 = nn.Linear(in_dim1, out_dim)
        self.proj2 = nn.Linear(in_dim2, out_dim)
        self.gate_linear = nn.Linear(out_dim*2, out_dim, bias=False)
    
    def forward(self, x1, x2):
        '''
        params:
            x1: [B, *, in_dim1]
            x2: [B, in_dim2]
        '''
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        if len(x1.size()) > 2:
            x2 = x2.unsqueeze(1).expand_as(x1)
        x = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.gate_linear(x))
        out = gate * x1 + (1-gate) * x2
        return out
    
class GatedFusion(nn.Module):
    def __init__(self, x_in_features, y_in_feautres, project_dim):
        super(GatedFusion, self).__init__()
        self.linear1 = nn.Linear(x_in_features, project_dim, bias=False)
        self.linear2 = nn.Linear(y_in_feautres, project_dim, bias=True)
    
    def forward(self, x, y):
        gate = F.sigmoid(self.linear1(x)+self.linear2(y))
        out = gate*x + (1-gate)*y
        return out

class GatedPromptFusion(nn.Module):
    def __init__(self, d_llm):
        super(GatedPromptFusion, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(d_llm, d_llm//4), nn.GELU(), 
                                     nn.Dropout(0.1), nn.Linear(d_llm//4, d_llm))
        self.linear2 = nn.Sequential(nn.Linear(d_llm, d_llm//4), nn.GELU(), 
                                     nn.Dropout(0.1), nn.Linear(d_llm//4, d_llm))
        self.projection = nn.Linear(d_llm, d_llm)
    
    def forward(self, x, y):
        gate = F.sigmoid(self.linear1(x)+self.linear2(y))
        out = gate*x + (1-gate)*y
        out = self.projection(out)
        return out