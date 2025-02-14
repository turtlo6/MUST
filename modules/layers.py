import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.utils import softmax

from torch.nn import Parameter
from torch_scatter import scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeightedGAT(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 drop_rate):
        super(WeightedGAT, self).__init__()

        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(drop_rate)
        self.ffd_drop = nn.Dropout(drop_rate)

        # self.residual = residual
        # if self.residual:
        #     self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, edge_index, edge_weight, feat):
        edge_weight = edge_weight.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(feat).view(-1, H, C)
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l

        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)

        coefficients = softmax(alpha, edge_index[1])

        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)

        x_j = x[edge_index[0]]

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)
        # if self.residual:
        #     out = out + self.lin_residual(feat)
        feat = out

        return feat

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)


class IGRU(Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(IGRU, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimension of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.reset_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.reset_bias = Parameter(torch.zeros(self.output_dim))
        self.act_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.act_bias = Parameter(torch.zeros(self.output_dim))
        self.update_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2 * self.input_dim, self.output_dim)))
        self.update_bias = Parameter(torch.zeros(self.output_dim))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.reset_wei)
        self.param.append(self.reset_bias)
        self.param.append(self.act_wei)
        self.param.append(self.act_bias)
        self.param.append(self.update_wei)
        self.param.append(self.update_bias)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, pre_state, cur_state):

        # ====================
        # Reset gate
        reset_input = torch.cat((cur_state, pre_state), dim=1)
        reset_output = torch.sigmoid(torch.mm(reset_input, self.param[0]) + self.param[1])
        # ==========
        # Input activation gate
        act_input = torch.cat((cur_state, torch.mul(reset_output, pre_state)), dim=1)
        act_output = torch.tanh(torch.mm(act_input, self.param[2]) + self.param[3])
        # ==========
        # Update gate
        update_input = torch.cat((cur_state, pre_state), dim=1)
        update_output = torch.sigmoid(torch.mm(update_input, self.param[4]) + self.param[5])
        # ==========
        # Next state
        next_state = torch.mul((1 - update_output), pre_state) + torch.mul(update_output, act_output)
        if self.training:
            next_state = self.dropout_layer(next_state)

        return next_state


class ILSTM(Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(ILSTM, self).__init__()
        # ====================
        self.input_dim = input_dim  # Dimensionality of input features
        self.output_dim = output_dim  # Dimension of output features
        self.dropout_rate = dropout_rate  # Dropout rate
        # ====================
        # Initialize model parameters
        self.input_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.input_bias = Parameter(torch.zeros(self.output_dim))
        self.forget_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.forget_bias = Parameter(torch.zeros(self.output_dim))
        self.cell_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.cell_bias = Parameter(torch.zeros(self.output_dim))
        self.output_wei = Init.xavier_uniform_(
            Parameter(torch.FloatTensor(self.input_dim + self.output_dim, self.output_dim)))
        self.output_bias = Parameter(torch.zeros(self.output_dim))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.input_wei)
        self.param.append(self.input_bias)
        self.param.append(self.forget_wei)
        self.param.append(self.forget_bias)
        self.param.append(self.cell_wei)
        self.param.append(self.cell_bias)
        self.param.append(self.output_wei)
        self.param.append(self.output_bias)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, pre_state, pre_cell, cur_input):

        # ====================
        combined = torch.cat((cur_input, pre_state), dim=1)

        # Input gate
        input_gate = torch.sigmoid(torch.matmul(combined, self.param[0]) + self.param[1])

        # Forget gate
        forget_gate = torch.sigmoid(torch.matmul(combined, self.param[2]) + self.param[3])

        # Cell candidate
        cell_candidate = torch.tanh(torch.matmul(combined, self.param[4]) + self.param[5])

        # Output gate
        output_gate = torch.sigmoid(torch.matmul(combined, self.param[6]) + self.param[7])

        # Next cell state
        next_cell = torch.mul(forget_gate, pre_cell) + torch.mul(input_gate, cell_candidate)

        # Next hidden state
        next_state = torch.mul(output_gate, torch.tanh(next_cell))
        if self.training:
            next_state = self.dropout_layer(next_state)

        return next_state, next_cell


class WeiPool(torch.nn.Module):

    def __init__(self):
        super(WeiPool, self).__init__()

    def forward(self, x, D):

        normD = D / D.sum()
        x = torch.sum(torch.matmul(normD, x), dim=0)
        return x


class MultiConcat(Module):

    def __init__(self):
        super(MultiConcat, self).__init__()

    def forward(self, micro_x, meso_x, macro_x):
        x = torch.cat([micro_x, meso_x, macro_x], dim=1)

        return x


class FCNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
