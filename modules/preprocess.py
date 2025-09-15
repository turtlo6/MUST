# _*_ coding : utf-8 _*_
# @Time : 2024/7/4 13:28
# @Author : wfr
# @file : preprocess
# @Project : IDEA

import torch
import pickle
from modules.utils import *
from modules.loss import *
import scipy.sparse
import random
import numpy as np
import networkx as nx
import torch_geometric as tg
from torch_geometric.data import Data
import scipy.sparse as sp
from community import community_louvain


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的CUDA随机种子（如果GPU可用）
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random库的随机种子
    torch.backends.cudnn.deterministic = True  # 设置使用CUDA加速时保证结果一致性


setup_seed(0)

# 15_25 20_30 30_40 35_45 40_50
# ================
data_name = 'GM_15_25_2000_10'
num_nodes = 100  # Number of nodes
num_snaps = 80  # Number of snapshots
# ================

edge_seq_list = np.load('../data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:80]

# =========
max_thres = 0  # Threshold for maximum edge weight
for i in range(len(edge_seq_list)):
    for j in range(len(edge_seq_list[i])):
        max_thres = max(edge_seq_list[i][j][2], max_thres)

# ==================
feat = np.load('../data/%s_feat.npy' % data_name, allow_pickle=True)
feat_list = []
for i in range(num_snaps):
    adj = get_adj_wei(edge_seq_list[i], num_nodes, max_thres)
    feat_list.append(np.concatenate((feat, adj), axis=1))

# ==========
pyg_graphs = []
for i in range(num_snaps):
    # ============
    edge_seq = edge_seq_list[i]
    adj = get_adj_norm_wei_with_self_loop(edge_seq, num_nodes, max_thres)
    adj_sp = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj_sp.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_sp.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj_normalized)
    # =============
    feat = feat_list[i]
    rowsum = np.array(feat.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feat = r_mat_inv.dot(feat)
    x = torch.FloatTensor(feat)
    # ==============
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    pyg_graphs.append(data)

# 保存
with open('../pyg_graphs/%s_pyg_graphs.pkl' % data_name, 'wb') as f:
    pickle.dump(pyg_graphs, f)

# =================
# =================
# =================
# =================

num_snaps = 80
num_nodes = 100
# =================

edge_seq_list = np.load('../data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:80]
# 转成合适的格式
edge_index_list = []
edge_weight_list = []
for i in range(num_snaps):
    # 去掉edge_seq中的边权重，并转为适合输入Node2Vec模型的格式
    edge_index = [[], []]
    edge_weight = []
    for edge in edge_seq_list[i]:  # 每条边代表的是无向边！！不存在重复
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        edge_weight.append(edge[2])  # 权重归一化
    # edge_index_tnr = torch.LongTensor(edge_index).to(device)
    # edge_weight_tnr = torch.FloatTensor(edge_weight).to(device)
    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)
data_name = '%s_80' % data_name
# ==================
# 创建nx格式的图列表
graphs = []
for edge_seq in edge_seq_list:
    # 创建一个新的无向图
    G = nx.Graph()
    # 添加节点特征
    for i in range(num_nodes):
        G.add_node(i)
    # 添加边和权重
    for edge in edge_seq:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight)
    # 将图添加到图列表中
    graphs.append(G)

# ===================
# 社团划分
partition_dict_list = []
for G in graphs:
    partition_dict = community_louvain.best_partition(G, random_state=1)  # key为节点编号，value为节点所属社团编号
    partition_dict_list.append(partition_dict)

# ===================
# 获取按社团划分的edge_index和edge_weight
edge_index_com_list_list = []  # 里面包含了窗口内每张图的每个社团的edge_index（每张图拥有多个edge_index，即每个社团的，组成一个列表）
edge_weight_com_list_list = []
for t in range(num_snaps):
    partition_dict = partition_dict_list[t]
    edge_index = edge_index_list[t]
    edge_weight = edge_weight_list[t]
    num_coms = max(partition_dict.values()) + 1  # 当前图的社团数量
    # ======
    edge_index_com_list = []  # 当前图的每个社团的edge_index的列表，列表长度等于社团数
    edge_weight_com_list = []
    for i in range(num_coms):
        edge_index_com_list.append([[], []])
        edge_weight_com_list.append([])
    # ======
    for i in range(len(edge_index[0])):  # 遍历所有边，看其端点是否属于同一社团，若属于则加入对应的edge_index_com
        node1 = edge_index[0][i]
        node2 = edge_index[1][i]
        weight = edge_weight[i]
        if partition_dict[node1] == partition_dict[node2]:
            com_id = partition_dict[node1]
            edge_index_com_list[com_id][0].append(node1)
            edge_index_com_list[com_id][1].append(node2)
            edge_weight_com_list[com_id].append(weight)
    # ==========
    # 为每个社团内的节点重新编号
    edge_index_com_list_new = []
    for com_id in range(num_coms):
        node_ids = [key for key, value in partition_dict.items() if value == com_id]
        node_set = set(node_ids)
        node_map = {node: i for i, node in enumerate(node_set)}  # 映射字典，旧编号到新编号的映射
        edge_index_com_new = [[node_map[node] for node in edge_index_com_list[com_id][0]],
                              [node_map[node] for node in edge_index_com_list[com_id][1]]]
        edge_index_com_list_new.append(edge_index_com_new)

    edge_index_com_list_list.append(edge_index_com_list_new)
    edge_weight_com_list_list.append(edge_weight_com_list)

# ========================
# np.save('../com_list_list/%s_edge_index_com_list_list.npy' % data_name, np.array(edge_index_com_list_list))
# np.save('../com_list_list/%s_edge_weight_com_list_list.npy' % data_name, np.array(edge_weight_com_list_list))
np.save('../com_list_list/%s_edge_index_com_list_list.npy' % data_name,
        np.array(edge_index_com_list_list, dtype=object))
np.save('../com_list_list/%s_edge_weight_com_list_list.npy' % data_name,
        np.array(edge_weight_com_list_list, dtype=object))