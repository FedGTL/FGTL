import logging
import os
import numpy as np
import torch
import pickle
import random
from itertools import permutations, combinations
from torch_scatter import scatter_add
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, to_networkx, degree, k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
from collections import Counter
import copy


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def knowledge_vote(knowledge_list, confidence_gate, num_classes, gpu):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """
    max_p, max_p_class = knowledge_list.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float().to(gpu)
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).to(gpu)
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    consensus_knowledge_mask = (max_conf > confidence_gate).float().to(gpu)
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).to(gpu).scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask


def calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers,
                              num_classes, gpu):
    """
    :param consensus_focus_dict: record consensus_focus for each domain
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :param source_domain_numbers: the numbers of source domains
    """
    domain_contribution = {frozenset(): 0}
    for combination_num in range(1, source_domain_numbers + 1):
        combination_list = list(combinations(range(source_domain_numbers), combination_num))
        for combination in combination_list:
            consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask = knowledge_vote(
                knowledge_list[:, combination, :], confidence_gate, num_classes, gpu=gpu)
            domain_contribution[frozenset(combination)] = torch.sum(
                consensus_knowledge_conf * consensus_knowledge_mask).item()
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (
                                                            domain_contribution[frozenset(
                                                                permutation[:permutation.index(source_idx) + 1])]
                                                            - domain_contribution[
                                                                frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict


def update_domain_weight(global_domain_weight, epoch_domain_weight, momentum=0.9):
    global_domain_weight = [round(global_domain_weight[i] * momentum + epoch_domain_weight[i] * (1 - momentum), 4)
                            for i in range(len(epoch_domain_weight))]
    return global_domain_weight


def federated_average(model_list, coefficient_matrix, batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """
    global_model_params = dict()

    if batchnorm_mmd:
        # running_mean and running_var in state_dict
        dict_list = [deserialize(it['classifier_params'] if 'classifier_params' in it else it['encoder_params']) for it in model_list]
        dict_item_list = [dic.items() for dic in dict_list]
        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]
            global_model_params[key_data_pair_list[0][0]] = sum(source_data_list)

    else:
        # no running_mean and running_var in named_parameters
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            global_model_params[parameter_list[0][0]] = sum(source_parameters)

    return global_model_params


def get_data(path):
    subgraphs, num_graphs, num_features, num_labels = pickle.load(
        open(os.path.join(path), "rb")
    )
    return subgraphs, num_graphs, num_features, num_labels


def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


class Log(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # console handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        fh.close()
        sh.close()

    def get_logger(self):
        return self.logger


class LPA:
    def __init__(self, args, cached: bool = True):
        self.args = args
        self.cached = cached
        self.cached_edge_index = None

    def normalize_adj(self, edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

        fill_value = 2. if improved else 1.
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if self.args.use_ppmi:
            adj_dict = {}

            def add_edge(a, b):
                if a in adj_dict:
                    neighbors = adj_dict[a]
                else:
                    neighbors = set()
                    adj_dict[a] = neighbors
                if b not in neighbors:
                    neighbors.add(b)

            cpu_device = torch.device("cpu")
            gpu_device = edge_index.device
            for a, b in edge_index.t().detach().to(cpu_device).numpy():
                a = int(a)
                b = int(b)
                add_edge(a, b)
                add_edge(b, a)

            adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

            def sample_neighbor(a):
                neighbors = adj_dict[a]
                random_index = np.random.randint(0, len(neighbors))
                return neighbors[random_index]


            # word_counter = Counter()
            walk_counters = {}

            def norm(counter):
                s = sum(counter.values())
                new_counter = Counter()
                for a, count in counter.items():
                    new_counter[a] = counter[a] / s
                return new_counter

            for _ in tqdm(range(40)):
                for a in adj_dict:
                    current_a = a
                    current_path_len = np.random.randint(1, self.args.path_len + 1)
                    for _ in range(current_path_len):

                        b = sample_neighbor(current_a)
                        if a in walk_counters:
                            walk_counter = walk_counters[a]
                        else:
                            walk_counter = Counter()
                            walk_counters[a] = walk_counter

                        walk_counter[b] += 1

                        current_a = b

            normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

            prob_sums = Counter()

            for a, normed_walk_counter in normed_walk_counters.items():
                for b, prob in normed_walk_counter.items():
                    prob_sums[b] += prob

            ppmis = {}

            for a, normed_walk_counter in normed_walk_counters.items():
                for b, prob in normed_walk_counter.items():
                    ppmi = max(np.log(prob / prob_sums[b] * len(prob_sums) / self.args.path_len), 0)
                    ppmis[(a, b)] = ppmi

            new_edge_index = []
            edge_weight = []
            for (a, b), ppmi in ppmis.items():
                new_edge_index.append([a, b])
                edge_weight.append(ppmi)

            edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
            edge_weight = torch.tensor(edge_weight).to(gpu_device)

            if add_self_loops :
                edge_index, edge_weight = add_remaining_self_loops(
                    edge_index, edge_weight, fill_value, num_nodes)

            row, col = edge_index
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            return edge_index, (edge_weight * deg_inv_sqrt[col]).type(torch.float32)
        
        else:
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                        device=edge_index.device)

            if add_self_loops:
                edge_index, tmp_edge_weight = add_remaining_self_loops(
                    edge_index, edge_weight, fill_value, num_nodes)
                assert tmp_edge_weight is not None
                edge_weight = tmp_edge_weight

            row, col = edge_index[0], edge_index[1]
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow_(-1)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            return edge_index, edge_weight * deg_inv_sqrt[col]


    def __call__(self, edge_index: torch.Tensor, y: torch.Tensor, mask):
        cache = self.cached_edge_index
        if cache is None:
            edge_index, edge_weight = self.normalize_adj(edge_index, None, num_nodes=y.shape[0])
            if self.cached:
                self.cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cache[0], cache[1]
        
        y_out = y.clone().detach()
        row, col = edge_index[0], edge_index[1]
        for i in range(self.args.num_iter):
            y_out = scatter_add(y_out[row] * edge_weight.unsqueeze(1), col, dim=0, dim_size=y.shape[0])
            y_out[mask] = y[mask]
        return y_out


def compute_ppmi(edge_index, edge_weight=None, num_nodes=None, path_len=5, improved=False,
             add_self_loops=True, dtype=None, gpu="cuda:0"):
    adj_dict = {}

    def add_edge(a, b):
        if a in adj_dict:
            neighbors = adj_dict[a]
        else:
            neighbors = set()
            adj_dict[a] = neighbors
        if b not in neighbors:
            neighbors.add(b)

    cpu_device = torch.device("cpu")
    gpu_device = torch.device(gpu)
    for a, b in edge_index.t().detach().to(cpu_device).numpy():
        a = int(a)
        b = int(b)
        add_edge(a, b)
        add_edge(b, a)

    adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

    def sample_neighbor(a):
        neighbors = adj_dict[a]
        random_index = np.random.randint(0, len(neighbors))
        return neighbors[random_index]


    # word_counter = Counter()
    walk_counters = {}

    def norm(counter):
        s = sum(counter.values())
        new_counter = Counter()
        for a, count in counter.items():
            new_counter[a] = counter[a] / s
        return new_counter

    for _ in tqdm(range(40)):
        for a in adj_dict:
            current_a = a
            current_path_len = np.random.randint(1, path_len + 1)
            for _ in range(current_path_len):
                b = sample_neighbor(current_a)
                if a in walk_counters:
                    walk_counter = walk_counters[a]
                else:
                    walk_counter = Counter()
                    walk_counters[a] = walk_counter

                walk_counter[b] += 1

                current_a = b

    normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

    prob_sums = Counter()

    for a, normed_walk_counter in normed_walk_counters.items():
        for b, prob in normed_walk_counter.items():
            prob_sums[b] += prob

    ppmis = {}

    for a, normed_walk_counter in normed_walk_counters.items():
        for b, prob in normed_walk_counter.items():
            ppmi = max(np.log(prob / prob_sums[b] * len(prob_sums) / path_len), 0)
            ppmis[(a, b)] = ppmi

    new_edge_index = []
    edge_weight = []
    for (a, b), ppmi in ppmis.items():
        new_edge_index.append([a, b])
        edge_weight.append(ppmi)

    edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
    edge_weight = torch.tensor(edge_weight).to(gpu_device)

    if add_self_loops :
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    return edge_index, edge_weight


def ppmi_norm(edge_index, edge_weight=None, num_nodes=None, path_len=5, improved=False,
             add_self_loops=True, dtype=None, gpu="cuda:0"):
    
    edge_index, edge_weight = compute_ppmi(edge_index, edge_weight, num_nodes, path_len, improved,
                                    add_self_loops, dtype, gpu)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x / 0.5)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def contrast_loss(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):

    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def AvgReadout(seq):
    return torch.mean(seq, 0)


def aug_random_mask(input_feature, drop_percent=0.2):
    
    node_num = input_feature.shape[0]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0])
    for j in mask_idx:
        aug_feature[j] = zeros
    return aug_feature


def aug_random_edge(input_feature, edge_index, drop_percent=0.2):

    node_num = input_feature.shape[0]
    _, edge_num = edge_index.size()
    permute_num = int(edge_num * drop_percent)

    edge_index = edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    new_edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    new_edge_index = torch.tensor(edge_index)

    return new_edge_index


def aug_drop_node(input_fea, edge_index, drop_percent=0.2):

    node_num = input_fea.shape[0]
    input_adj = torch.zeros(node_num, node_num, dtype=torch.long)
    input_adj[edge_index[0], edge_index[1]] = 1

    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_adj = torch.nonzero(aug_input_adj).t()

    return aug_input_fea, aug_input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


def _convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tpl in enumerate(graph_infos):
        idx, x = tpl[0].edge_index[0], tpl[0].x
        deg = degree(idx, tpl[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tpl[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def to_egoNet(graph, ego, hop_number): 
    # get ego-networks for sampled nodes
    sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(ego, hop_number, graph.edge_index)
    
    def re_index(source):
        mapping = dict(zip(sub_nodes.numpy(), range(sub_nodes.shape[0])))
        return mapping[source]
    
    edge_index_u = [*map(re_index, sub_edge_index[0][:].numpy())]
    edge_index_v = [*map(re_index, sub_edge_index[1][:].numpy())]

    egonet = Data(edge_index=torch.tensor([edge_index_u, edge_index_v]), x=graph.x[sub_nodes], y=graph.y[sub_nodes])
    return egonet

def get_egonetworks(graph, ego_number, hop_number, sampling):
    ego_number = min(ego_number, graph.num_nodes)
    
    num_graphs = ego_number
    num_features = graph.num_node_features
    num_labels = len(graph.y.unique())

    if sampling == 'random':
        # egos = []
        # batch_size = graph.num_nodes // ego_number
        # for i in range(ego_number):
        #     egos += random.sample(range(batch_size * i, batch_size * (i+1)), 1)
        egos = random.sample(range(graph.num_nodes), ego_number)
        print("random ego central nodes:{}".format(egos))
        egonetworks = [to_egoNet(graph, ego, hop_number) for ego in egos]

    if sampling == 'byLabel':
        egos_byLabel = {}
        allLabels = graph.y.unique()
        for label in allLabels:
            idx_label = np.where(graph.y == label)[0]
            egos_byLabel[label.item()] = random.sample(list(idx_label), ego_number)    # ego_number is per client in this case (should be smaller)

        egonetworks = {k: [to_egoNet(graph, ego.item(), hop_number) for ego in v] for k, v in egos_byLabel.items()}
        num_graphs = len(allLabels) * ego_number

    if (sampling == 'random' and not egonetworks[0].__contains__('x')):
        egonetworks = _convert_to_nodeDegreeFeatures(egonetworks)

    if (sampling == 'byLabel' and not list(egonetworks.values())[0].__contains__('x')):
        egonetworks = {k: _convert_to_nodeDegreeFeatures(v) for k, v in egonetworks.items()}


    return egonetworks


class GradReverse(torch.autograd.Function):
    rate = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg()*GradReverse.rate
        return grad_output, None


class GRL(nn.Module):
    @staticmethod
    def forward(inp):
        return GradReverse.apply(inp)


def filter_by_specified_keywords(param_name, filter_keywords):
    '''
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    '''
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve


def param_filter(state_dict, filter_keywords=None):
    '''
    Filter some of the parameters in the model parameters

    Arguments:
        state_dict (dict): PyTorch Module object's state_dict.
    Returns:
        state_dict (dict): remove the keys that match any of the given
        keywords.
    '''
    return dict(filter(
                lambda elem: filter_by_specified_keywords(elem[0], filter_keywords),
                state_dict.items()
            ))


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    shuffle_index = None
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def batch_ppmi(num_nodes, shuffle_index, edge_index, edge_weight):
    a = torch.zeros((num_nodes, num_nodes)).double().to(edge_weight.device)
    a[edge_index[0], edge_index[1]] = edge_weight
    a = a[shuffle_index, :][:, shuffle_index]

    # row norm in each mini-batch
    rowsum = torch.sum(a, dim=-1)
    r_inv = rowsum.pow(-1)
    r_inv.masked_fill_(r_inv == float('inf'), 0.)
    r_mat_inv = torch.diag(r_inv)
    r_norm_a = torch.matmul(r_mat_inv, a)
    return r_norm_a


def calculate_parameters(model: nn.Module):
    # compute the parameter load of the model
    total = sum([param.numel() for param in model.parameters()])
    return total