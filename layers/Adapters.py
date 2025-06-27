import torch
from torch import nn
class Adapter(nn.Module):
    '''
    Adapater is a 2-layer FFN with a bottleneck
    '''
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim), 
                                 nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, input_dim))
    
    def forward(self, x):
        out = self.mlp(x)
        out = out + x
        return out

class PBTtLayerWithAdapter(nn.Module):
    def __init__(self, configs, original_layer, adapter_size=32):
        super(PBTtLayerWithAdapter, self).__init__()
        # Copy all parameters and submodules from original layer
        self.original_layer = original_layer
        self.adapter = Adapter(configs.d_model, adapter_size)
        
    def forward(self, *args, **kwargs):
        # Run original layer
        out, guide_loss, LB_loss  = self.original_layer(*args, **kwargs)
        # Run the adapter
        out = self.adapter(out)
        return out, guide_loss, LB_loss 

