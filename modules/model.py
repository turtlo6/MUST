import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from .layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLP(Module):

    def __init__(self, micro_dims, agg_feat_dim, RNN_dims, decoder_dims,
                 n_heads,
                 dropout_rate):
        super(MultiLP, self).__init__()

        self.micro_dims = micro_dims  # dimension of the GAT layer for learning micro-representation
        self.agg_feat_dim = agg_feat_dim
        self.RNN_dims = RNN_dims
        self.decoder_dims = decoder_dims
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        # ===================
        # 学习微观表示
        self.num_micro_GAT_layers = len(self.micro_dims) - 1
        self.micro_GAT_layers = nn.ModuleList()
        for l in range(self.num_micro_GAT_layers):
            self.micro_GAT_layers.append(
                WeightedGAT(input_dim=self.micro_dims[l], output_dim=self.micro_dims[l + 1], n_heads=self.n_heads,
                            drop_rate=self.dropout_rate))

        self.macro_pooling_layers = WeiPool()
        self.meso_pooling_layers = WeiPool()
        self.agg_layers = MultiConcat()

        self.num_RNN_layers = len(self.RNN_dims) - 1
        self.RNN_layers = nn.ModuleList()
        for l in range(self.num_RNN_layers):
            self.RNN_layers.append(
                ILSTM(input_dim=self.RNN_dims[l], output_dim=self.RNN_dims[l + 1], dropout_rate=self.dropout_rate))
        # ==================

        self.decoder = FCNN(self.decoder_dims[0], self.decoder_dims[1], self.decoder_dims[2], self.dropout_rate)

    def forward(self, edge_index_list, edge_weight_list, feat_list, D_com_list_list, partition_dict_list,
                D_list,
                pred_flag=True):

        win_size = len(feat_list)
        num_nodes = feat_list[0].shape[0]
        # =======================
        input_micro_feat_list = feat_list
        output_micro_feat_list = None
        for l in range(self.num_micro_GAT_layers):
            micro_layer = self.micro_GAT_layers[l]
            output_micro_feat_list = []
            for t in range(win_size):
                # if l == 0 and t == 9:
                #     print(1)
                output_micro_feat = micro_layer(edge_index_list[t], edge_weight_list[t], input_micro_feat_list[t])
                output_micro_feat_list.append(output_micro_feat)
            input_micro_feat_list = output_micro_feat_list
        # =======================
        output_meso_feat_list = []
        for t in range(win_size):
            partition_dict = partition_dict_list[t]
            output_micro_feat = output_micro_feat_list[t]
            D_com_list = D_com_list_list[t]
            output_meso_feat = torch.empty(num_nodes, self.micro_dims[-1]).to(device)
            for com_idx in range(len(D_com_list)):
                cur_com_nodes_list = [key for key, value in partition_dict.items() if
                                      value == com_idx]
                D_com = D_com_list[com_idx]

                output_meso_com_feat = self.meso_pooling_layers(output_micro_feat[cur_com_nodes_list], D_com)
                output_meso_feat[cur_com_nodes_list] = output_meso_com_feat
            output_meso_feat_list.append(output_meso_feat)
        # =======================
        output_macro_feat_list = []
        for t in range(win_size):
            output_macro_feat = self.macro_pooling_layers(output_micro_feat_list[t], D_list[t])
            output_macro_feat = output_macro_feat.expand(num_nodes, -1)  # 扩充列数使其形状与微观表示矩阵相同
            output_macro_feat_list.append(output_macro_feat)
        # =======================
        output_agg_feat_list = []
        for t in range(win_size):
            output_agg_feat = self.agg_layers(output_micro_feat_list[t], output_meso_feat_list[t],
                                              output_macro_feat_list[t])
            output_agg_feat_list.append(output_agg_feat)
        # ======================
        input_RNN_list = output_agg_feat_list
        # input_RNN_list = output_micro_feat_list
        output_RNN_list = None
        for l in range(self.num_RNN_layers):
            RNN_layer = self.RNN_layers[l]
            output_RNN_list = []
            pre_state = torch.zeros(num_nodes, self.RNN_dims[l + 1]).to(device)
            pre_cell = torch.zeros(num_nodes, self.RNN_dims[l + 1]).to(device)
            for t in range(win_size):
                output_state, output_cell = RNN_layer(pre_state, pre_cell, input_RNN_list[t])
                pre_state = output_state
                pre_cell = output_cell
                output_RNN_list.append(output_state)
            input_RNN_list = output_RNN_list
        # ======================
        if pred_flag:  # prediction mode
            input_feat = F.normalize(output_RNN_list[-1], dim=0, p=2)
            # input_feat = output_RNN_list[-1]
            pred_adj = self.decoder(input_feat)
            return [pred_adj]
        else:  # training mode
            pred_adj_list = []
            for t in range(win_size):
                input_feat = F.normalize(output_RNN_list[t], dim=0, p=2)
                # input_feat = output_RNN_list[t]
                pred_adj = self.decoder(input_feat)
                pred_adj_list.append(pred_adj)
            return pred_adj_list
        # ============
