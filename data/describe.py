import pandas as pd
import torch
import torch_geometric
import torch_geometric.utils as U
from tabulate import tabulate


def describe(g : torch_geometric.data.Data, infos=['basic', 'label', 'degree'], deg_types=None):
    if 'degree' in infos and deg_types is None:
        deg_types = ['both']
    
    
    if 'basic' in infos:
        df = get_basic_df(g)
        print('The basic information of the graph:')
        print(tabulate(df, headers='keys', tablefmt ='psql', showindex=False))
    if 'label' in infos:
        df = get_labeled_df(g)
        print('The top label information of the graph:')
        print(tabulate(df.head(5), headers='keys', tablefmt ='psql', floatfmt=".2%", showindex=False))
        print('The bottom label information of the graph:')
        print(tabulate(df.tail(5), headers='keys', tablefmt ='psql', floatfmt=".2%", showindex=False))
    if 'degree' in infos:
        for deg_type in deg_types:
            df = get_degree_df(g, deg_type)
            print(f'The {deg_type}-degree information of the graph:')
            print(tabulate(df, headers='keys', tablefmt ='psql', floatfmt=".2%", showindex=False))



def get_basic_df(g):
    table = [
        ['The number of nodes:', g.num_nodes],
        ['The number of node features:', g.num_node_features],
        ['The number of edges:', g.num_edges],
        ['The number of class:', len(g.y.unique())],
        ['The number of train nodes:', g.train_mask.sum().item()],
        ['The number of val nodes:', g.val_mask.sum().item()],
        ['The number of test nodes:', g.test_mask.sum().item()],
        ['Avg degree:', round(g.num_edges / g.num_nodes)],
        ['Is a symetric graph:', g.is_undirected()],
        ['has_isolated_nodes:', g.has_isolated_nodes()],
        ['Contains self loop:', g.has_self_loops()],
        ['Contains duplicate edges:', g.is_coalesced()],
    ]
    df = pd.DataFrame(table, columns=['', 'Statistics'])
    return df


def get_labeled_df(g):
    label_total = g.y.shape[0]
    table = [
        ['# labeled nodes:', label_total, label_total / g.num_nodes],
    ]
    label_cnts = torch.bincount(g.y.flatten())
    for i in range(len(torch.unique(label_cnts))):
        table.append([f'# label - {i}', label_cnts[i].item(), label_cnts[i].item() / label_total])
    df = pd.DataFrame(table, columns=['', 'cnt', 'pct'])
    df = df.sort_values('pct', ascending=False)
    return df

def get_degree_df(g, deg_type='in'):
    assert deg_type in ['in', 'out', 'both']
    if deg_type == 'in':
        degrees = U.degree(g.edge_index[1], num_nodes=g.num_nodes).int()
    elif deg_type == 'out':
        degrees = U.degree(g.edge_index[0], num_nodes=g.num_nodes).int()
    elif deg_type == 'both':
        degrees = U.degree(g.edge_index.flatten(), num_nodes=g.num_nodes).int()
    degree_cnt = torch.bincount(degrees)
    # degree_cnt = torch.scatter_add(torch.zeros(degrees.max()+1, dtype=int), 0, degrees.long(), torch.ones_like(degrees, dtype=int))
    degree_cumsum = torch.cumsum(degree_cnt, dim=0)
    table = []
    for i in range(11):
        table.append(
            [
                f'The number of nodes with {i}-{deg_type}-degrees:', 
                degree_cnt[i].item(), 
                degree_cnt[i].item() / degree_cumsum[-1].item(), 
                degree_cumsum[i].item(), degree_cumsum[i].item() / degree_cumsum[-1].item()]
        )
    df = pd.DataFrame(table, columns=['', 'cnt', 'pct', 'cum_cnt', 'cum_pct'])
    return df

