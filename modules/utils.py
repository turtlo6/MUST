import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score


def get_adj_wei(edges, num_nodes, max_wei):
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > max_wei:
            wei = max_wei
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj


def get_adj_norm_wei_with_self_loop(edges, num_nodes, max_wei):
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > max_wei:
            wei = max_wei
        adj[src, dst] = wei / max_wei
        adj[dst, src] = wei / max_wei
    for i in range(num_nodes):
        adj[i, i] = 0.5

    return adj


def get_adj_no_wei(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei > 0:
            wei = 1
        else:
            wei = 0
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj


def get_AUC(pred_adj, gnd):
    return roc_auc_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)))


def get_PRAUC(pred_adj, gnd):
    return average_precision_score(np.reshape(gnd, (-1,)), np.reshape(pred_adj, (-1,)))


def get_D_by_edge_index_and_weight(edge_index, edge_weight, num_nodes):
    D = np.zeros((num_nodes, num_nodes))
    for i in range(len(edge_index[0])):
        node1 = edge_index[0][i]
        node2 = edge_index[1][i]
        wei = edge_weight[i]
        D[node1, node1] += wei
        D[node2, node2] += wei

    # only one node within the community
    if len(edge_index[0]) == 0:
        D[0, 0] = 1.

    return D


def get_D_by_edge_index_and_weight_tnr(edge_index, edge_weight, num_nodes):
    D = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index[0].shape[0]):
        node1 = edge_index[0][i].item()
        node2 = edge_index[1][i].item()
        wei = edge_weight[i].item()
        if node1 != node2:
            D[node1, node1] += wei

    if edge_index[0].shape[0] == 1:
        D[0, 0] = 1.

    return D
