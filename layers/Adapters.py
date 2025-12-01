import torch
from torch import nn
class Adapter(nn.Module):
    '''
    Adapater is a 2-layer FFN with a bottleneck
    '''
    def __init__(self, input_dim, hidden_dim, use_norm=True):
        super(Adapter, self).__init__()
        if use_norm:
            self.mlp = nn.Sequential(nn.LayerNorm(input_dim), 
                                    nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, input_dim))
        else:
            self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, input_dim))
    
    def forward(self, x):
        out = self.mlp(x)
        out = out + x
        return out

class PBTCPLayerWithAdapter(nn.Module):
    def __init__(self, configs, original_layer, adapter_size=32):
        super(PBTCPLayerWithAdapter, self).__init__()
        # Copy all parameters and submodules from original layer
        self.bottom_adapter = Adapter(3*configs.charge_discharge_length, adapter_size, use_norm=False)
        self.original_layer = original_layer
        self.top_adapter = Adapter(configs.d_model, adapter_size)
        
    def forward(self, cycle_curve_data, DKP_embeddings, total_masks, ion_type_masks, use_view_experts):
        out = self.bottom_adapter(cycle_curve_data) # [B, L, 3*charge_discharge_length]
        # Run original layer
        out, guide_loss, LB_loss = self.original_layer(cycle_curve_data, DKP_embeddings, total_masks, ion_type_masks=ion_type_masks, use_view_experts=use_view_experts)
        # Run the adapter
        out = self.top_adapter(out)
        return out, guide_loss, LB_loss 


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


class CPLayerWithAdapter(nn.Module):
    def __init__(self, configs, original_layer, adapter_size=32):
        super(CPLayerWithAdapter, self).__init__()
        # Copy all parameters and submodules from original layer
        self.bottom_adapter = Adapter(configs.charge_discharge_length, adapter_size, use_norm=False)
        self.original_layer = original_layer
        self.top_adapter = Adapter(3*configs.charge_discharge_length, adapter_size)
        
    def forward(self, cycle_curve_data):
        out = self.bottom_adapter(cycle_curve_data) # [B, L, 3*charge_discharge_length]
        # Run original layer
        out = self.original_layer(cycle_curve_data)
        # Run the adapter
        out = self.top_adapter(out)
        return out 


class tLayerWithAdapter(nn.Module):
    def __init__(self, configs, original_layer, adapter_size=32):
        super(tLayerWithAdapter, self).__init__()
        # Copy all parameters and submodules from original layer
        self.original_layer = original_layer
        self.adapter = Adapter(configs.d_model, adapter_size)
        self.model = configs.model
        
    def forward(self, *args, **kwargs):
        # Run original layer
        out = self.original_layer(*args, **kwargs)
        
        # Run the adapter
        out = self.adapter(out)
        return out 

class CPTtLayerWithAdapter(nn.Module):
    def __init__(self, configs, original_layer, adapter_size=32):
        super(CPTtLayerWithAdapter, self).__init__()
        # Copy all parameters and submodules from original layer
        self.original_layer = original_layer
        self.adapter = Adapter(configs.d_model, adapter_size)
        
    def forward(self, *args, **kwargs):
        # Run original layer
        out, attn = self.original_layer(*args, **kwargs)
        
        # Run the adapter
        out = self.adapter(out)
        return out, attn

