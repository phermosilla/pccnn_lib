import torch
from torch_scatter import scatter_mean

class GroupNormalization(torch.nn.Module):

    def __init__(self, p_num_features, p_num_groups):

        # Super class init.
        super(GroupNormalization, self).__init__()

        self.num_features_ = p_num_features
        self.num_groups_ = p_num_groups

        # Create the parameters.
        self.gamma_ = torch.nn.Parameter(
            torch.empty(1, p_num_features))
        self.gamma_.data.fill_(1.0)

        self.betas_ = torch.nn.Parameter(
            torch.empty(1, p_num_features))
        self.betas_.data.fill_(0.0)
    

    def forward(self, p_feats, p_pc):
        eps_val = 1e-8
        group_size = self.num_features_//self.num_groups_

        cur_feats = p_feats.reshape((-1, self.num_groups_))
        cur_batch_ids = p_pc.batch_ids_.reshape((-1, 1)).\
            repeat(1, group_size).to(torch.int64).reshape((-1,))

        feat_batch_means = scatter_mean(cur_feats, cur_batch_ids, dim=0)
        feat_means = torch.index_select(feat_batch_means, 0, cur_batch_ids)

        feat_stddevs = (cur_feats - feat_means)**2
        feat_batch_stddevs = scatter_mean(feat_stddevs, cur_batch_ids, dim=0)
        feat_stddevs = torch.index_select(feat_batch_stddevs, 0, cur_batch_ids)

        cur_feats = (cur_feats - feat_means)/torch.sqrt(feat_stddevs + eps_val)
        cur_feats = cur_feats.reshape((-1, self.num_features_))
        return cur_feats*self.gamma_ + self.betas_
        