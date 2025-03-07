import torch
import torch.nn as nn
import torch.nn.functional as F

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