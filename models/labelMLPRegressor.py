import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.d_llm = args.d_llm
        self.d_ff = args.d_ff
        self.align_layer = nn.Sequential(nn.Linear(self.d_llm, self.d_ff), nn.ReLU())
        self.head = nn.Linear(self.d_ff, args.output_num)
    
    def forward(self, label_prompt_embedding):
        features = self.align_layer(label_prompt_embedding)
        pred = self.head(features)
        pred = pred.float()
        return pred, features