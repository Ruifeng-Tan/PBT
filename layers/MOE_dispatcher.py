'''
This script is modified from https://github.com/decisionintelligence/DUET/blob/main/ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py
'''
import torch
from torch import nn


# class MOEDispatcher(nn.Module):
#     def __init__(self, num_experts, gates, top_K):
#       super(MOEDispatcher, self).__init__()
#       self.top_K = top_K
#       self.num_experts = num_experts
#       self._gates = gates
#       # sort experts
#       sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#       # drop indices
#       _, self._expert_index = sorted_experts.split(1, dim=1)
#       # get according batch index for each expert
#       self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
#       self._part_sizes = (gates > 0).sum(0).tolist()
#       # expand gates to match with self._batch_index
#       gates_exp = gates[self._batch_index.flatten()]
#       self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

#     def dispatch(self):
#       '''
#       Return:
#           MOE_indices: a list of Tensor whose shape is [expert_batch_size]
#       '''
#       return torch.split(self._batch_index, self._part_sizes, dim=0)

#     def combine(self, expert_out, multiply_by_gates=True):
#       """Sum together the expert output, weighted by the gates.
#       The slice corresponding to a particular batch element `b` is computed
#       as the sum over all experts `i` of the expert output, weighted by the
#       corresponding gate values.  If `multiply_by_gates` is set to False, the
#       gate values are ignored.
#       Args:
#         expert_out: a list of `num_experts` `Tensor`s, each with shape
#           `[expert_batch_size_i, <extra_output_dims>]`.
#         multiply_by_gates: a boolean
#       Returns:
#         a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
#       """
#       stitched = torch.cat(expert_out, 0)
#       if multiply_by_gates:
#           # stitched = stitched.mul(self._nonzero_gates)
#           stitched = torch.einsum('i...,ij->i...', stitched, self._nonzero_gates)

#       shape = list(expert_out[-1].shape)
#       shape[0] = self._gates.size(0)
#       zeros = torch.zeros(*shape, requires_grad=True,
#                           device=expert_out[-1].device)
#       # combine samples that have been processed by the same k experts
#       combined = zeros.index_add(0, self._batch_index, stitched.float())
#       return combined
    
#     def rearrange_tensor(self, stitched):
#       '''
#       Args:
#         stitched: [2B, *]
#       Retruns:
#         Y: [B, K, *]. K is the number of acivated experts (Top-K).
#       '''
#       # Get the batch size B
#       B = len(self._batch_index) // 2
      
#       # Create an index tensor to rearrange X
#       # This will have shape [B, 2] where each row contains indices of the two representations
#       indices = torch.zeros((B, 2), dtype=torch.long)
      
#       # Fill in the indices array
#       for i in range(B):
#           indices[i] = torch.where(self._batch_index == i)[0]
      
#       # Use advanced indexing to gather the pairs
#       Y = stitched[indices]
    
#       return Y
    
class MOEDispatcher(nn.Module):
    def __init__(self, num_experts, gates):
        super(MOEDispatcher, self).__init__()
        self.num_experts = num_experts
        self._gates = gates
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self):
        '''
        Return:
            MOE_indices: a list of Tensor whose shape is [expert_batch_size]
        '''
        return torch.split(self._batch_index, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = torch.einsum('i...,ij->i...', stitched, self._nonzero_gates)

        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True,
                            device=expert_out[-1].device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined