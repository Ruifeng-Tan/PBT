import torch
import torch.nn as nn
class PatternRouter_two(nn.Module):
    def __init__(self, input_size1, input_size2, d_ff, num_experts):
        super(PatternRouter_two, self).__init__()
        self.input2_linear = nn.Sequential(nn.Linear(input_size2, d_ff), nn.ReLU(), nn.Linear(d_ff, input_size1))
        self.pattern_fit = nn.Sequential(nn.Linear(input_size1, num_experts, bias=True))

    def forward(self, input1, input2):
        '''
        input1: [B, size1]
        input2: [B, size2]
        '''
        input2 = self.input2_linear(input2)
        x = input1 + input2
        out = self.pattern_fit(x)
        return out

class PatternRouter_three(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, d_ff, num_experts):
        super(PatternRouter_three, self).__init__()
        self.pattern_fit = nn.Linear(input_size1+input_size2+input_size3, num_experts, bias=True)
        #self.pattern_fit = nn.Sequential(nn.Linear(input_size1+input_size2+input_size3, num_experts))

    def forward(self, input1, input2, input3):
        '''
        input1: [B, size1]
        input2: [B, size2]
        input3: [B, size3]
        '''
        x = torch.cat([input1, input2, input3], dim=-1)
        out = self.pattern_fit(x)
        return out

class PatternRouter_Imp_three(nn.Module):
    def __init__(self, DKP_size, cycle_number_size, fine_info_size, d_ff, num_experts):
        super(PatternRouter_Imp_three, self).__init__()
        self.DKP_linear = nn.Linear(DKP_size, d_ff)
        self.cycle_number_linear = nn.Linear(cycle_number_size, d_ff)
        self.fine_info_linear = nn.Linear(fine_info_size, d_ff)
        self.distribution_fit = nn.Sequential(nn.Linear(d_ff*3, num_experts, bias=True))

    def forward(self, DKP, cycle_number, fine_info):
        DKP = self.DKP_linear(DKP)
        cycle_number = self.cycle_number_linear(cycle_number)
        fine_info = self.fine_info_linear(fine_info)
        x = torch.cat([DKP, cycle_number, fine_info], dim=1) # [B, d_ff*2]
        out = self.distribution_fit(x)
        return out
    
class PatternRouter_Imp(nn.Module):
    def __init__(self, input_size, fine_info_size, d_ff, num_experts):
        super(PatternRouter_Imp, self).__init__()
        self.x_linear = nn.Linear(input_size, d_ff)
        self.fine_info_linear = nn.Linear(fine_info_size, d_ff)
        self.distribution_fit = nn.Sequential(nn.Linear(d_ff*2, num_experts, bias=True))

    def forward(self, x, fine_info):
        x = self.x_linear(x)
        fine_info = self.fine_info_linear(fine_info)
        x = torch.cat([fine_info, x], dim=1) # [B, d_ff*2]
        out = self.distribution_fit(x)
        return out
    
class PatternRouter(nn.Module):
    def __init__(self, input_size, num_experts):
        super(PatternRouter, self).__init__()
        self.distribution_fit = nn.Sequential(nn.Linear(input_size, num_experts, bias=True))

    def forward(self, x):
        out = self.distribution_fit(x)
        return out


class MergedPatternRouter(nn.Module):
    def __init__(self, n_vars, early_cycle_threshold, charge_discharge_length, d_ff, d_llm, num_experts):
        super(MergedPatternRouter, self).__init__()
        input_size = n_vars * charge_discharge_length
        input_size2 = d_ff * early_cycle_threshold
        self.flattn1 = nn.Flatten(start_dim=2)
        self.linear1 = nn.Linear(input_size, d_ff, bias=False)

        self.flattn2 = nn.Flatten(start_dim=1)
        self.linear2 = nn.Linear(input_size2, d_ff, bias=False)

        self.distribution_fit = nn.Sequential(nn.Linear(d_llm+d_ff, d_ff, bias=False), nn.ReLU(),
                                              nn.Linear(d_ff, num_experts, bias=False))
        
    def forward(self, x, DKP):
        '''
        x: [B, L, 3, fixed_len]
        DKP: [B, d_llm]
        '''
        x = self.flattn1(x) # [B, L, 3*fixed_len]
        out = self.linear1(x) # [B, L, d_ff]

        out = self.flattn2(out) # [B, L*d_ff]
        out = self.linear2(out) # [B, d_ff]

        out = torch.cat([out, DKP], dim=1)
        out = self.distribution_fit(out)
        return out

class DistributionRouter(nn.Module):
    def __init__(self, config):
        super(DistributionRouter, self).__init__()
        input_size = config.d_llm
        num_experts = config.num_experts
        encoder_hidden_size = config.d_ff

        # self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
        #                                       nn.Linear(encoder_hidden_size, num_experts, bias=False))
        self.distribution_fit = nn.Sequential(nn.Linear(input_size, num_experts, bias=True))

    def forward(self, x):
        out = self.distribution_fit(x)
        return out

class ReLURouter(nn.Module):
    """Route each token to the experts with non-zero relu outputs."""

    def __init__(self, configs, input_size) -> None:
        """Initialize the relu router.

        Args:
            configs
        """
        super(ReLURouter, self).__init__()
        self.topk = configs.topK
        self.num_experts = configs.num_experts
        self.pattern_fit = nn.Sequential(nn.Linear(input_size, self.num_experts), nn.ReLU())
        self.relu = nn.ReLU()
    
    def forward(self, x):
        '''
        x: [B, input_size]
        '''
        B = x.shape[0]
        logits = self.relu(self.pattern_fit(x))
        # compute the sparsity
        activation_density = 0
        if self.training:
            activation_density = torch.count_nonzero(logits) / (B*self.num_experts)
        
        return logits, activation_density
