import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding
from typing import Optional
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, battery_life_config):
        super(Model, self).__init__()
        configs = battery_life_config.ec_config.get_configs()
        self.tokenizer = None
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length*3, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])

        self.pe = PositionalEmbedding(self.d_model)
        self.inter_TransformerEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ]
        )
        self.down_sample_ratio = configs.down_sample_ratio
        self.dropout = nn.Dropout(configs.dropout)
        self.inter_flatten = nn.Flatten(start_dim=1)
        self.projection = nn.Sequential(nn.Linear(self.d_model, configs.output_num))

    def forward(self, cycle_curve_data, curve_attn_mask, 
                attention_mask: Optional[torch.Tensor] = None,
                DKP_embeddings: Optional[torch.FloatTensor] = None,
                cathode_masks: Optional[torch.Tensor] = None,
                temperature_masks: Optional[torch.Tensor] = None,
                format_masks: Optional[torch.Tensor] = None,
                anode_masks: Optional[torch.Tensor] = None,
                ion_type_masks: Optional[torch.Tensor] = None,
                combined_masks: Optional[torch.Tensor] = None,
                SOH_trajectory: Optional[torch.Tensor] = None,
                CE_trajectory: Optional[torch.Tensor] = None,
                use_aug: bool = False,
                return_embedding: bool=False):
        '''
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        '''
        B, L, num_var, fixed_len = cycle_curve_data.shape[0], cycle_curve_data.shape[1], cycle_curve_data.shape[2], cycle_curve_data.shape[3]
        if use_aug:
            random_point_num =  int(fixed_len*self.down_sample_ratio)

            flatten_cycle_curve_data = cycle_curve_data.reshape(B, L*num_var, -1)
            flatten_cycle_curve_data = flatten_cycle_curve_data.transpose(1, 2) # [B, fixed_len, L*num_var]

            flatten_cycle_curve_data = flatten_cycle_curve_data.expand(3, -1, -1, -1)  # [3, B, fixed_len, L*num_var]
            flatten_cycle_curve_data = flatten_cycle_curve_data.reshape(3 * B, fixed_len, flatten_cycle_curve_data.shape[-1])  # [3*B, fixed_len, L*num_var]
            flatten_cycle_curve_data[B:] = self.CD_Crop_augmentation(flatten_cycle_curve_data[B:], random_point_num) # add noise
            flatten_cycle_curve_data = flatten_cycle_curve_data.transpose(1, 2) # [2*B, L*num_var, fixed_len]
            cycle_curve_data = flatten_cycle_curve_data.reshape(3*B, L, num_var, fixed_len)

            # cycle_curve_data = torch.cat([cycle_curve_data, aug_cycle_curve_data], dim=0)
            curve_attn_mask = curve_attn_mask.unsqueeze(0).expand(3, -1, -1).reshape(3*B, -1)
        
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, early_cycle, fixed_len * num_var]
        cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, early_cycle, d_model]

        cycle_curve_data = self.pe(cycle_curve_data) + cycle_curve_data
        attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        attn_mask = torch.repeat_interleave(attn_mask, attn_mask.shape[-1], dim=1) # [B, L, L]
        attn_mask = attn_mask.unsqueeze(1) # [B, 1, L, L]
        attn_mask = attn_mask==0 # set True to mask
        output, attns = self.inter_TransformerEncoder(cycle_curve_data, attn_mask=attn_mask)

        lengths = torch.sum(curve_attn_mask, dim=1).cpu() # [N]
        idx = (torch.as_tensor(lengths, device=output.device, dtype=torch.long) - 1).view(-1, 1).expand(
            len(lengths), output.size(2))
        idx = idx.unsqueeze(1)
        output = output.gather(1, idx).squeeze(1) # [B, D]

        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, L * d_model)
        preds = self.projection(output)  # (batch_size, num_classes)
        
        return preds[:B], None, output[B:], None, None, None, 0, 0

    def CD_Crop_augmentation(self, X: torch.Tensor, random_point_num: int) -> torch.Tensor:
        """
        Resamples the input tensor X by selecting random points (including start and end) and interpolating back to original length.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, L, D].
            random_point_num (int): Number of points to randomly select, must be at least 2.
        
        Returns:
            torch.Tensor: Resampled tensor of shape [B, L, D].
        """
        B, L, D = X.shape
        charge_X = X[:, :L//2]
        discharge_X = X[:, L//2:]
        aug_charge_X = self.Crop(charge_X, random_point_num=random_point_num//2)
        aug_discharge_X = self.Crop(discharge_X, random_point_num=random_point_num//2)

        X = torch.cat([aug_charge_X, aug_discharge_X], dim=1) # [B, random_point_num, D]
        return X

    def Crop(self, X: torch.Tensor, random_point_num: int) -> torch.Tensor:
        """
        Resamples the input tensor X by selecting random points (including start and end) and interpolating back to original length.
        
        Args:
            X (torch.Tensor): Input tensor of shape [B, L, D].
            random_point_num (int): Number of points to randomly select, must be at least 2.
        
        Returns:
            torch.Tensor: Resampled tensor of shape [B, L, D].
        """
        B, L, D = X.shape
        K = random_point_num

        if K < 2:
            raise ValueError("random_point_num must be at least 2")
        if L < 2:
            raise ValueError("L must be at least 2")

        # Step 1: Generate indices including 0 and L-1, plus random middle points
        middle = L - 2  # Possible middle indices count (1 to L-2)
        k_middle = K - 2

        if k_middle > 0:
            if middle < k_middle:
                raise ValueError(f"Cannot select {k_middle} middle points when middle={middle}")

            # Generate random middle indices without replacement
            rand_vals = torch.rand(B, middle, device=X.device)
            middle_indices = rand_vals.argsort(dim=1)[:, :k_middle]  # [B, k_middle]
            middle_indices += 1  # Shift to range 1 to L-2 (inclusive)
        else:
            raise Exception(f'random_point_num is {random_point_num}. You should set it larger than 2!')

        # Combine with start and end indices
        zeros = torch.zeros(B, 1, dtype=torch.long, device=X.device) # start indices
        ends = (L - 1) * torch.ones(B, 1, dtype=torch.long, device=X.device) # end indices
        indices = torch.cat([zeros, middle_indices, ends], dim=1)
        indices, _ = torch.sort(indices, dim=1)  # Sort indices for each batch

        # Step 2: Gather the selected points
        selected_X = torch.gather(X, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        selected_X = selected_X.transpose(1, 2) # [B, D, random_point_num]

        # Step 3: do the linear interpolation
        interpolated = F.interpolate(
                selected_X,
                size=L,
                mode='linear',
                align_corners=True
            )
        
        return interpolated.transpose(1, 2)