import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.fusion import GatedFusion
from layers.MLPs import MLPBlockGELU, PBTMLPBlock
from layers.distributional_router_encoder import PatternRouterMLP
from layers.MOE_dispatcher import MOEDispatcher

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class RMSEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(RMSEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

    
class MoEEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=8, top_k=2, num_general_experts=1, charge_discharge_length=300):
        super(MoEEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.drop_rate = dropout
        self.num_experts = num_experts
        self.num_general_experts = num_general_experts
        self.charge_discharge_length = charge_discharge_length
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.gate  = PatternRouterMLP(self.d_model, self.num_experts)
        self.experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(num_experts)])
        self.general_experts = nn.ModuleList([MLPBlockGELU(self.d_model, self.d_ff, self.drop_rate, self.activation) for _ in range(self.num_general_experts)])
        self.norm2 = nn.LayerNorm(self.d_model)
        self.eps = 1e-9

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''
        params:
            x: [B, L, d_model]
            attn_mask: [B, L, L]. Set True to mask
        '''
        B, L = x.shape[0], x.shape[1]

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        x = x.reshape(B*L, -1)
        logits = self.gate(x)
        _, indices = torch.topk(logits, self.top_k, dim=1)
        # Create a mask where only the top-K values will be kept
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the mask at the indices of the top-K values
        mask.scatter_(1, indices, 1) # 0 indicates mask
        logits = F.softmax(logits, dim=1) # [B, num_experts]
        raw_logits = logits.clone()
        # logits.masked_fill_(mask==0, 0) # [B, num_experts]
        logits = logits * mask
        de_norm = torch.sum(logits, dim=1) + self.eps
        logits = logits / de_norm.unsqueeze(-1)

        dispatcher = MOEDispatcher(self.num_experts, logits)
        MOE_indicies = dispatcher.dispatch()
        total_outs = []
        total_expert_outs = []
        for i, expert in enumerate(self.experts):
            out = expert(x[MOE_indicies[i]]) # [expert_batch_size, d_llm]
            total_outs.append(out)
            if len(MOE_indicies[i])>=1:
                total_expert_outs.append(out)

        total_outs = dispatcher.combine(total_outs).to(torch.bfloat16) # [B*L, d_model]
        final_out = total_outs
        for i in range(self.num_general_experts):
            final_out = self.general_experts[i](x) + final_out
        out = self.norm2(final_out + x) # add & norm
        out = out.reshape(B, L, -1)

        aug_loss = 0
        if self.training:
            # Compute the auxiliary loss
            expert_logits = torch.mean(raw_logits, dim=0) # [num_experts]
            expert_sample_count = torch.count_nonzero(logits, dim=0) / (B*L) # [num_experts]
            aug_loss = torch.sum(expert_logits * expert_sample_count) # [1]

        return out, attn, aug_loss


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
    
class MoEEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MoEEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        total_LB_count = 0
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn, aug_loss = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
                total_LB_count += aug_loss

            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, aug_loss = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                total_LB_count += aug_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, total_LB_count


class PBTEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, gate_dim, d_ff=None, dropout=0.1, activation="relu"):
        super(PBTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.mlp = PBTMLPBlock(d_model, d_ff, gate_dim, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None, DKP_embeddings=None, knowledge_mask=None):
        x = self.norm1(x) # pre-attention norm
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm2(x) # post-attention norm

        y = self.mlp(y, DKP_embeddings, knowledge_mask)

        return x + y, attn
    
class PBTEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(PBTEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, DKP_embeddings=None, knowledge_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, DKP_embeddings=DKP_embeddings, knowledge_mask=knowledge_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, DKP_embeddings=DKP_embeddings, knowledge_mask=knowledge_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class RMSDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(RMSDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


# class MyDecoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(MyDecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask,
#             tau=tau, delta=None
#         )[0])
#         x = self.norm1(x)

#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask,
#             tau=tau, delta=delta
#         )[0])

#         y = x = self.norm2(x)
#         y = self.activation(self.conv1(y.transpose(-1, 1)))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm3(x + y)
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


