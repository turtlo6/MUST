import torch
import pickle
import torch.optim as optim
import torch_geometric as tg
from modules.model import MultiLP
from modules.utils import *
from modules.loss import *
import scipy.sparse
import random
import numpy as np
import networkx as nx
from community import community_louvain
import datetime
import os

device = torch.device('cuda')


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)

data_name = 'GM_15_25_2000_10'
num_nodes = 100  # Number of nodes
num_snaps = 80  # Number of snapshots
feat_dim = 132  # Dimensionality of node feature
GAT_output_dim = 44
micro_dims = [feat_dim, 100, GAT_output_dim]
pooling_ratio = 0.8
agg_feat_dim = GAT_output_dim * 3
RNN_dims = [agg_feat_dim, 256, 256]
decoder_dims = [RNN_dims[-1], 256, num_nodes]
save_flag = False

# =================
dropout_rate = 0.5  # Dropout rate
win_size = 10  # Window size of historical snapshots
num_epochs = 800  # Number of training epochs
num_test_snaps = 20  # Number of test snapshots
num_train_snaps = num_snaps - num_test_snaps  # Number of training snapshots
n_heads = 2
# =================
step_interval = 5
early_stop_epochs = 50
# =================
# loss
lambd_reg = 0.001
theta = 0.5  # Decaying factor
sparse_beta = 10

# =================
edge_seq_list = np.load('data/%s_edge_seq.npy' % data_name, allow_pickle=True)
edge_seq_list = edge_seq_list[0:80]
# =================
with open('pyg_graphs/%s_pyg_graphs.pkl' % data_name, 'rb') as f:
    pyg_graphs = pickle.load(f)
D_list = []
edge_index_list = []
edge_weight_list = []
feat_list = []
for i in range(num_snaps):
    pyg_graph = pyg_graphs[i].to(device)
    edge_index = pyg_graph.edge_index
    edge_weight = pyg_graph.edge_weight
    feat = pyg_graph.x
    D = get_D_by_edge_index_and_weight_tnr(pyg_graph.edge_index, pyg_graph.edge_weight, num_nodes).to(device)
    D_list.append(D)
    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)
    feat_list.append(feat)

# ================
data_name = '%s_80' % data_name
# ==================
graphs = []
for edge_seq in edge_seq_list:
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for edge in edge_seq:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight)
    graphs.append(G)

# ===================
partition_dict_list = []
for G in graphs:
    partition_dict = community_louvain.best_partition(G, random_state=1)
    partition_dict_list.append(partition_dict)

# ==================
edge_index_com_list_list = np.load('com_list_list/%s_edge_index_com_list_list.npy' % data_name, allow_pickle=True)
edge_weight_com_list_list = np.load('com_list_list/%s_edge_weight_com_list_list.npy' % data_name,
                                    allow_pickle=True)
D_com_list_list = []
for i in range(len(edge_index_com_list_list)):
    D_com_list = []
    edge_index_com_list = edge_index_com_list_list[i]
    edge_weight_com_list = edge_weight_com_list_list[i]
    for j in range(len(edge_index_com_list)):
        num_nodes_com = 0
        for key, value in partition_dict_list[i].items():
            if value == j:
                num_nodes_com += 1
        D_com = get_D_by_edge_index_and_weight(edge_index_com_list[j], edge_weight_com_list[j], num_nodes_com)
        D_com_tnr = torch.FloatTensor(D_com).to(device)
        D_com_list.append(D_com_tnr)
    D_com_list_list.append(D_com_list)
# ==================
model = MultiLP(micro_dims, agg_feat_dim, RNN_dims, decoder_dims, n_heads, dropout_rate).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

# ==================
best_AUC = 0.
best_PRAUC = 0.
no_improve_epochs = 0
for epoch in range(num_epochs):
    # ============
    model.train()
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    # ============
    train_cnt = 0
    loss_list = []
    # =======================
    random.seed(epoch)
    indices = list(range(win_size, num_train_snaps))
    random.shuffle(indices)
    # =======================
    iteration_count = 0
    for tau in indices:
        # ================
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ===================
        cur_D_list = D_list[tau - win_size: tau]
        cur_D_com_list_list = D_com_list_list[tau - win_size: tau]
        # ================
        gnd_list = []
        for t in range(tau - win_size + 1, tau + 1):
            edges = edge_seq_list[t]
            gnd = get_adj_no_wei(edges, num_nodes)
            gnd_tnr = torch.FloatTensor(gnd).to(device)
            gnd_list.append(gnd_tnr)
        # ================
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                              cur_D_com_list_list, cur_partition_dict_list, cur_D_list, pred_flag=False)
        iteration_count += 1
        loss = get_reg_loss(sparse_beta, gnd_list, pred_adj_list, theta, lambd_reg)
        loss.backward()

        if iteration_count % step_interval == 0:
            opt.step()
            opt.zero_grad()
            iteration_count = 0

        # ===============
        loss_list.append(loss.item())
        train_cnt += 1
    if iteration_count % step_interval != 0:
        opt.step()
        opt.zero_grad()
    loss_mean = np.mean(loss_list)
    print('Epoch#%d Train G-Loss %f' % (epoch, loss_mean))

    if save_flag:
        torch.save(model, 'my_pt/MUST_%s_%d.pkl' % (data_name, epoch))

    # =====================
    model.eval()
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    # =============
    AUC_list = []
    PRAUC_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    for tau in range(num_snaps - num_test_snaps, num_snaps):
        # ================
        # Prediction data for the current window
        cur_edge_index_list = edge_index_list[tau - win_size: tau]
        cur_edge_weight_list = edge_weight_list[tau - win_size: tau]
        cur_feat_list = feat_list[tau - win_size: tau]
        cur_partition_dict_list = partition_dict_list[tau - win_size: tau]
        # ===================
        cur_D_list = D_list[tau - win_size: tau]
        cur_D_com_list_list = D_com_list_list[tau - win_size: tau]
        # ================
        pred_adj_list = model(cur_edge_index_list, cur_edge_weight_list, cur_feat_list,
                              cur_D_com_list_list, cur_partition_dict_list, cur_D_list, pred_flag=True)
        pred_adj = pred_adj_list[-1]
        # ===========================
        pred_adj = pred_adj.cpu().data.numpy()
        for r in range(num_nodes):
            pred_adj[r, r] = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                pre_av = (pred_adj[i, j] + pred_adj[j, i]) / 2
                pred_adj[i, j] = pre_av
                pred_adj[j, i] = pre_av
        # ==============
        edges = edge_seq_list[tau]
        gnd = get_adj_no_wei(edges, num_nodes)
        # ===============
        # Calculate the evaluation indicators for the current time step
        AUC = get_AUC(pred_adj, gnd)
        PRAUC_new = get_PRAUC(pred_adj, gnd)
        # ===============
        AUC_list.append(AUC)
        PRAUC_list.append(PRAUC_new)
    # ==============
    AUC_mean = np.mean(AUC_list)
    AUC_std = np.std(AUC_list, ddof=1)
    PRAUC_mean = np.mean(PRAUC_list)
    PRAUC_std = np.std(PRAUC_list, ddof=1)

    if AUC_mean <= best_AUC:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_epochs:
            break
    else:
        best_AUC = AUC_mean
        best_PRAUC = PRAUC_mean
        no_improve_epochs = 0
    # ==============
    print(
        'Test AUC %f %f PRAUC %f %f best_AUC %f best_PRAUC %f'
        % (AUC_mean, AUC_std, PRAUC_mean, PRAUC_std, best_AUC, best_PRAUC))
    # ==========
    f_input = open('res/MUST_%s.txt' % data_name, 'a+')
    f_input.write(
        'Test AUC %f %f PRAUC %f %f best_AUC %f best_PRAUC %f Time %s\n'
        % (AUC_mean, AUC_std, PRAUC_mean, PRAUC_std, best_AUC, best_PRAUC, current_time))
    f_input.close()
