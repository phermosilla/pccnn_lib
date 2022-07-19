import torch
 
class Bias(torch.nn.Module):
     
    def __init__(self, p_num_features):

       super(Bias, self).__init__()
       self.bias_ = torch.nn.Parameter(torch.empty(1, p_num_features))
       self.bias_.data.fill_(0.0)

    def forward(self, p_in_features):
        return p_in_features + self.bias_