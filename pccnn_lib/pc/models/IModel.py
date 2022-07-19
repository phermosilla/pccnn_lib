import numpy as np
import torch
import pcconv
import math 

class IModel:

    def __init__(self):
        self.conv_list_ = []

    def add_conv(self, p_conv):
        self.conv_list_.append(p_conv)

    def get_num_convs(self):
        return len(self.conv_list_)

    def set_init_warmup_state(self, p_conv_id, p_w_state):
        self.conv_list_[p_conv_id].set_init_warmup_state(p_w_state)

    def init_variance(self, p_conv_id, p_divisor):
        cur_var = p_divisor * self.conv_list_[p_conv_id].feat_input_size_
        stdv = math.sqrt(self.conv_list_[p_conv_id].out_constant_variance_/cur_var)
        self.conv_list_[p_conv_id].conv_weights_.data.normal_(0.0, stdv)
        if isinstance(self.conv_list_[p_conv_id], pcconv.pc.layers.PointConv):
            self.conv_list_[p_conv_id].init_proj_axis_ = True