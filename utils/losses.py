"""
Loss functions for PyTorch.
"""
import torch
import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        if self.similarity_type == 'l2':
            # Compute negative L2 distance
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'dot':
            # Compute dot product similarity
            return torch.matmul(features, features.transpose(0, 1))
        elif self.similarity_type == 'cosine':
            # Compute cosine similarity
            # Normalize the features to unit vectors
            features_normalized = F.normalize(features, p=2, dim=-1)
            return torch.matmul(features_normalized, features_normalized.transpose(0, 1))
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

class WeightedRnCLoss(nn.Module):
    '''
    Revised from https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py#L34
    - We only use the samples whose label diff is larger than the pos label diff as negative samples
    - The similarities of the negative pairs are weighted according to their label diff to the anchor sample
    2023NIPS Rank-N-Contrast: Learning Continuous Representations for Regression
    '''
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(WeightedRnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [2*bs, feat_dim]. features from augmented views
        # labels: [bs, label_dim]

        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = torch.repeat_interleave(labels, dim=0, repeats=2)  # [2bs, label_dim]
        labels = labels.repeat(2, 1)

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits -= logits_max.detach()
        exp_logits = logits.exp()
        # print(exp_logits)

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        valid_count = 0  # Count of samples with valid negative samples
        # print(logits)
        for k in range(n - 1):
            pos_logits = logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs > pos_label_diffs.view(-1, 1)).float()
            # neg_mask[:, k] = 1.0 # include the sample itself into the negative pairs to keep the training stability

            # neg_sample_num = neg_mask.sum(dim=-1)
            # neg_weight_logits = label_diffs - pos_label_diffs.view(-1, 1)
            # neg_weight_logits = neg_weight_logits.masked_fill_(neg_mask == 0, -np.inf)
            # # neg_weights = F.softmax(neg_weight_logits, dim=-1) * neg_sample_num.unsqueeze(-1)
            # neg_weights = F.softmax(neg_weight_logits, dim=-1)
            zero_neg_mask = (neg_mask.sum(dim=-1) == 0)
            sum_neg_mask = (neg_mask * exp_logits).mean(dim=-1)
            
            pos_log_probs = torch.zeros_like(pos_logits)
            if (~zero_neg_mask).any():
                pos_log_probs[~zero_neg_mask] = pos_logits[~zero_neg_mask] - torch.log(sum_neg_mask[~zero_neg_mask])
                valid_count += (~zero_neg_mask).sum().item()

            # Only add to loss for samples with valid negatives
            loss += -pos_log_probs[~zero_neg_mask].sum()

        # Normalize the loss by the number of valid samples
        if valid_count > 0:
            loss /= valid_count
        return loss

class RnCLoss(nn.Module):
    '''
    Revised from https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py#L34
    2023NIPS Rank-N-Contrast: Learning Continuous Representations for Regression
    '''
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [2*bs, feat_dim]. features from augmented views
        # labels: [bs, label_dim]

        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = torch.repeat_interleave(labels, dim=0, repeats=2)  # [2bs, label_dim]
        labels = labels.repeat(2, 1)

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits -= logits_max.detach()
        exp_logits = logits.exp()
        # print(exp_logits)

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        valid_count = 0  # Count of samples with valid negative samples
        # print(logits)
        for k in range(n - 1):
            pos_logits = logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs > pos_label_diffs.view(-1, 1)).float()

            # Check if there are negative samples
            sum_neg_mask = (neg_mask * exp_logits).sum(dim=-1)
            zero_neg_mask = (sum_neg_mask == 0)

            # Compute the log probabilities only for samples with negative samples
            pos_log_probs = torch.zeros_like(pos_logits)
            if (~zero_neg_mask).any():
                pos_log_probs[~zero_neg_mask] = pos_logits[~zero_neg_mask] - torch.log(sum_neg_mask[~zero_neg_mask])
                valid_count += (~zero_neg_mask).sum().item()  # Count valid samples
            loss += - pos_log_probs[~zero_neg_mask].sum()
        # Normalize the loss by the number of valid samples
        if valid_count > 0:
            loss /= valid_count

        return loss
    
class domain_averaged_MSELoss(nn.Module):
    def __init__(self):
        super(domain_averaged_MSELoss, self).__init__()
    
    def forward(self, outputs, labels, domain_ids):
        # Compute squared errors for each sample
        squared_errors = (outputs - labels) ** 2  # Shape [B]
        
        # Get unique domain IDs and their corresponding group indices
        unique_domains, group_indices = torch.unique(domain_ids, return_inverse=True)
        num_domains = unique_domains.size(0)

        # Sum squared errors per domain using scatter_add
        sum_se = torch.zeros(num_domains, device=squared_errors.device, dtype=squared_errors.dtype)
        sum_se.scatter_add_(0, group_indices, squared_errors)
        
        # Count the number of samples per domain using scatter_add
        counts = torch.zeros(num_domains, device=squared_errors.device, dtype=torch.long)
        counts.scatter_add_(0, group_indices, torch.ones_like(group_indices, dtype=torch.long))
        
        # Compute MSE for each domain and avoid division by zero (though counts are at least 1 here)
        mse_per_domain = sum_se / counts.float()
        
        # Average the MSEs across all domains
        loss = mse_per_domain.mean()
        
        return loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
def bmc_loss(pred, target, noise_var, reduce):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=target.device), reduce=reduce)     # contrastive-like loss
    # loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class Alignment_loss(nn.Module):
    def __init__(self, temperature=1.0, instance_alingment_weight=1.0):
        super(Alignment_loss, self).__init__()
        self.beta = instance_alingment_weight
        self.tau = temperature
    
    def forward(self, x, aug_x, label_prompt_embedding):
        D = x.shape[-1]
        # center-level alignment
        pos_dist = torch.norm(x-label_prompt_embedding, p=2, dim=1) / self.tau / np.sqrt(D) # [N]
        distance_matrix = torch.norm(x.unsqueeze(1) - label_prompt_embedding.unsqueeze(0), p=2, dim=-1) / self.tau / np.sqrt(D) # [N, N]
        pos_dist, distance_matrix = torch.exp(pos_dist), torch.exp(distance_matrix)
        
        distance_matrix = torch.sum(distance_matrix, dim=1) # [N]
        center_alignment_loss = torch.mean(torch.log(pos_dist / distance_matrix))

        # instance-level alignment
        pos_dist = torch.norm(x-aug_x, p=2, dim=1) / self.tau / np.sqrt(D) # [N]
        distance_matrix = torch.norm(x.unsqueeze(1) - aug_x.unsqueeze(0), p=2, dim=-1) / self.tau / np.sqrt(D) # [N, N]
        pos_dist, distance_matrix = torch.exp(pos_dist), torch.exp(distance_matrix)
        distance_matrix = torch.sum(distance_matrix, dim=1) # [N]
        instance_alignment_loss = torch.mean(torch.log(pos_dist / distance_matrix))

        alignment_loss = center_alignment_loss + self.beta * instance_alignment_loss
        return alignment_loss, center_alignment_loss, instance_alignment_loss



class DG_loss(nn.Module):
    def __init__(self, temperature=1.0, cl_weight=0.5):
        super(DG_loss, self).__init__()
        self.tau = temperature
        self.cl_weight = cl_weight
    
    def forward(self, features, class_indices, class_centers):
        '''
        features: [2N, d_llm]
        class_indices: [N]
        class_centers: [class_num, d_llm]
        '''
        N = features.shape[0]//2
        aug_features = features[N:]
        features = features[:N]

        selected_center_vectors = torch.index_select(class_centers, dim=0, index=class_indices) # [N, d_llm]
        # The embeddings of each sample should be close to its class embedding
        norm_loss = torch.norm(features-selected_center_vectors, p=2, dim=-1) # [N]
        norm_loss = torch.mean(norm_loss) # [1]

        # contrastive learning to avoid model collapse
        # cosine similarity is used here
        features = F.normalize(features, p=2, dim=-1)
        aug_features = F.normalize(aug_features, p=2, dim=-1)
        pos_sim = torch.cosine_similarity(features, aug_features, dim=1) / self.tau # [N]
        pos_sim = torch.exp(pos_sim) # [N]

        sim_matrix = torch.matmul(features, features.transpose(0,1)) / self.tau # [N, N]
        sim_matrix = torch.exp(sim_matrix)
        sim_matrix = torch.sum(sim_matrix, dim=1) # [N]

        cl_loss = - torch.mean(torch.log(pos_sim / sim_matrix)) # [1]
        loss = norm_loss + self.cl_weight * cl_loss
        return loss

class Battery_life_alignment_CL_loss(nn.Module):
    '''
    Contrastive learning loss for battery life prediction
    '''
    def __init__(self, args, dist_threshold, DG_weight, max_mean, max_std, cl_loss_weight=0.01, instance_tau=1, cluster_tau=1):
        super(Battery_life_alignment_CL_loss, self).__init__()
        self.args = args
        self.dist_threshold = dist_threshold
        self.DG_weight = DG_weight
        self.pdist = nn.PairwiseDistance(p=2)
        self.instance_tau = instance_tau
        # self.cluster_tau = cluster_tau
        self.cl_loss_weight = cl_loss_weight
        self.max_mean = max_mean
        self.max_std = max_std
    


    def forward(self, features, cl_embed, center_vectors, parametric_centers, class_labels, curve_attn_mask):
        '''
        features: [2*N, L, D]
        cl_embed: [2*N, d_ff]
        center_vectors: [token_num, D] the center vectors from the text modality
        parametric_centers: [num_class, D] the centers for each life class
        class_labels: [N] the life class labels
        curve_attn_mask: [2N, L] 0 means the position is masked
        '''
        N, L, D = features.shape[0]//2, features.shape[1], features.shape[-1]
        curve_attn_mask = curve_attn_mask[:N]
        token_num = center_vectors.shape[0]
        # raw_features = features.clone()
        # center_vectors = F.normalize(center_vectors, dim=-1, p=2)
        # features = F.normalize(features, dim=-1, p=2)
        aug_features = features[N:]
        anchor_features = features[:N]
        # alignment loss
        # Align the modality between the pretrained token embeddings and learned token embeddings
        # The embeddings for the padding tokens are ignored since they are masked in attention
        # If DG is used, the parametric centers will also be aligned to the LLM space
        if self.args.use_align:
            center_vectors = center_vectors.unsqueeze(0)
            center_vectors = center_vectors.expand(N, -1, -1)
            tmp_anchor_features = anchor_features.unsqueeze(2) # [N, L, 1, D]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.unsqueeze(2) # [1, num_class, 1, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1, -1) # [N, num_class, 1, D]
                tmp_anchor_features = torch.cat([tmp_anchor_features, tmp_parametric_centers], dim=1) # [N, L+num_class, 1, D]

            tmp_anchor_features = tmp_anchor_features.expand(-1, -1, token_num, -1) # [N, L, token_num, D]
            tmp_center_vectors = center_vectors.unsqueeze(1) # [N, 1, token_num, D]
            tmp_center_vectors = tmp_center_vectors.expand(-1, tmp_anchor_features.shape[1], -1, -1) # [N, L, token_num, D]
            # We only hope that the scale of two modality are aligned
            center_mean = torch.mean(tmp_center_vectors, dim=-1) # [N, L, token_num]
            anchor_mean = torch.mean(tmp_anchor_features, dim=-1) # [N, L, token_num]

            center_std = torch.std(tmp_center_vectors, dim=-1)
            anchor_std = torch.std(tmp_anchor_features, dim=-1)


            # dist = (anchor_mean-center_mean)*(anchor_mean-center_mean) # squared error w.r.t mean
            # dist += (anchor_std-center_std)*(anchor_std-center_std) # squared error w.r.t std
            mean_dist = F.relu(torch.abs(anchor_mean - center_mean) - self.max_mean)
            std_dist = F.relu(torch.abs(anchor_std - center_std) - self.max_std)
            dist = mean_dist + std_dist
            curve_attn_mask = curve_attn_mask.unsqueeze(-1).expand(-1,-1,token_num) # [N, L, token_num]
            if self.args.DG:
                tmp_parametric_centers = parametric_centers.unsqueeze(0) # [1, num_class, D]
                tmp_parametric_centers = tmp_parametric_centers.expand(N, -1, -1) # [N, num_class, D]
                parametric_center_masks = torch.ones_like(tmp_parametric_centers[:,:,0]) # [N, num_class]
                parametric_center_masks = parametric_center_masks.unsqueeze(-1).expand(-1,-1,token_num) # [N, num_class, token_num]
                curve_attn_mask = torch.cat([curve_attn_mask, parametric_center_masks], dim=1)

            dist = torch.where(curve_attn_mask==1, dist, torch.zeros_like(dist)) # [N, L, token_num]
            dist = dist.reshape(N, -1) # [N, L*toekn_num]
            curve_attn_mask = curve_attn_mask.reshape(N, -1)  # [N, L*toekn_num]
            dist = torch.sum(dist, dim=1) / torch.sum(curve_attn_mask, dim=1) # [N]
            # alignment_loss = F.relu(dist-self.dist_threshold) # loss with a margin
            alignment_loss = dist
            alignment_loss = torch.mean(alignment_loss) # 1

        # use contrastive learning 
        aug_features = cl_embed[N:] # [N, d_llm]
        anchor_features = cl_embed[:N]  # [N, d_llm]

        pos_sim = - self.pdist(anchor_features, aug_features) # [N]
        pos_sim = torch.exp(pos_sim/self.instance_tau) # [N]
        # Sim between anchor and negative sample
        tmp_features = cl_embed[:, None].expand(-1, N, -1) # [2N, N, D]
        tmp_anchor_features = anchor_features[:,None].expand(-1, 2*N, -1) # [N, 2N, D]

        neg_sim = - torch.norm(tmp_anchor_features - tmp_features.transpose(0,1), dim=2, p=2) # [N, 2N]
        neg_sim = torch.exp(neg_sim/self.instance_tau) # [N, 2N]
        if not self.args.DG:
            neg_sim = torch.sum(neg_sim, dim=1) # [N]
            cl_loss = - torch.mean(torch.log(pos_sim / neg_sim))
            return_cl_loss = cl_loss
            DG_loss = 0
        else:
            # use DG
            # add domain generalization loss
            # positive samples
            anchor_features = cl_embed[:N] # [N, d_llm]
            selected_centers = torch.index_select(parametric_centers, dim=0, index=class_labels)
            DG_pos_sim = - self.pdist(anchor_features, selected_centers) # [N]
            DG_pos_sim = torch.exp(DG_pos_sim/self.instance_tau) # [N]

            # negative samples
            anchor_features = anchor_features[:,None].expand(-1,N,-1) # [N, N, d_llm]
            selected_centers = selected_centers[:,None].expand(-1,N,-1) # [N, N, d_llm]

            DG_neg_sim = - torch.norm(anchor_features - selected_centers.transpose(0,1), dim=2, p=2) # [N, N]
            DG_neg_sim = torch.exp(DG_neg_sim/self.instance_tau) # [N, N]
            neg_sim = torch.cat([neg_sim, DG_neg_sim], dim=1) # [N, 3N]
            neg_sim = torch.sum(neg_sim, dim=1) # N

            DG_term = torch.log(DG_pos_sim / neg_sim) # [N]
            CL_term = torch.log(pos_sim / neg_sim) # [N]

            DG_loss = - torch.mean(DG_term) # 1
            return_cl_loss = - torch.mean(CL_term) # 1
            cl_loss = - torch.mean(DG_term + CL_term)

        if self.args.use_align:
            loss = alignment_loss + self.cl_loss_weight * cl_loss
        else:
            loss = self.cl_loss_weight * cl_loss
            alignment_loss = 0

        return loss, alignment_loss, return_cl_loss, DG_loss
    
class Battery_life_CL_loss(nn.Module):
    '''
    Contrastive learning loss for battery life prediction
    '''
    def __init__(self, cluster_cl_loss_weight=5, instance_tau=1, cluster_tau=1):
        super(Battery_life_CL_loss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.instance_tau = instance_tau
        self.cluster_tau = cluster_tau
        self.cluster_cl_loss_weight = cluster_cl_loss_weight
    


    def forward(self, features, center_vectors, center_vectors_indices):
        '''
        features: [N, D]
        center_vectors_indices: [N]
        center_vectors: [center_num, D]
        '''
        N = features.shape[0]//2
        center_vectors = F.normalize(center_vectors, dim=1, p=2)
        features = F.normalize(features, dim=1, p=2)
        aug_features = features[features.shape[0]//2:] # [N, D]
        features = features[:features.shape[0]//2] # [N, D]

        # Instance-wise contrastive learning
        # Anchor with positive samples
        pos_dist = self.pdist(features, aug_features) / self.instance_tau # [N]
        pos_dist = torch.exp(pos_dist) # [N]

        # Anchor with negative samples
        sim_m = torch.norm(features[:, None]-features, dim=2, p=2) / self.instance_tau # [N, N]
        sim_m = torch.exp(sim_m) # [N, N]
        denominator = torch.sum(sim_m, dim=1) # [N]

        instance_wise_cl_loss = torch.mean(torch.log(pos_dist / denominator))

        # Cluster-wise constrastive learning
        # Anchor with its center

        # anchor_centers = center_vectors[center_vectors_indices] # [N, D]
        anchor_centers = torch.index_select(center_vectors, dim=0, index=center_vectors_indices)
        pos_sim = torch.cosine_similarity(features, anchor_centers, dim=1) / self.cluster_tau # [N]
        pos_sim = torch.exp(pos_sim)

        # Anchor with other centers
        features = F.normalize(features, p=2, dim=1) # [N,D]
        center_vectors = F.normalize(center_vectors, p=2, dim=1) # [center_num,D]
        sim_m = torch.mm(features, center_vectors.transpose(0,1)) # [N, center_num]
        sim_m = sim_m / self.cluster_tau  # [N, center_num]
        sim_m = torch.exp(sim_m)
        denominator = torch.sum(sim_m, dim=1) # [N]

        cluster_wise_cl_loss = - torch.mean(torch.log(pos_sim / denominator))


        cl_loss = instance_wise_cl_loss + self.cluster_cl_loss_weight * cluster_wise_cl_loss

        return cl_loss   
        
        
        
        
       

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
