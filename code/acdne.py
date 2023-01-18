import os
import copy
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import ACDNE_model, NetworkEmbedding, models
from torchmetrics.functional import accuracy, f1_score
from torch_geometric.loader import NeighborLoader


class Server(object):
    def __init__(self, args, model: nn.Module):
        self.args = args
        self.model = model
        self.current_round_parames = []
        self.filter_keywords = ['domain_discriminator']

    def send_global_model(self):
        agg_model_para = param_filter(self.model.state_dict(), self.filter_keywords)
        return copy.deepcopy(serialize(agg_model_para))

    def receive_parameters_info(self, paraminfo):
        self.current_round_parames.append(paraminfo)

    def aggregation(self):

        total_params = {}
        n_total_samples = 0
        for paraminfo in self.current_round_parames:
            n_samples = paraminfo['n_samples']
            
            for k, v in deserialize(paraminfo['params']).items():
                if k not in total_params:
                    total_params[k] = []
                total_params[k].append(v.cpu().numpy() * n_samples)

            n_total_samples += n_samples

        parameters = {}
        for k, v in total_params.items():
            parameters[k] = torch.tensor(v).sum(dim=0) / n_total_samples
        self.model.load_state_dict(parameters, strict=False)
        
        self.current_round_parames.clear()

    def federated_average(self, param_list: List):
        '''
        the aggregation of clients' Network_Embedding

        :param param_list: record source client and target client model parameter
        '''
        avg_params = dict()
        source_samples = param_list[0]['n_samples']
        target_samples = param_list[1]['n_samples']
        n_total_samples = source_samples + target_samples
        source_param = deserialize(param_list[0]['params'])
        target_param = deserialize(param_list[1]['params'])
        
        for key in source_param:
            if key in target_param.keys():
                avg_params[key] = (source_samples * source_param[key] + target_samples * target_param[key]) / n_total_samples
            else:
                avg_params[key] = source_param[key].clone()

        paraminfo = dict(
            n_samples = n_total_samples,
            params = serialize(avg_params),
        )

        return paraminfo


class Client(object):
    def __init__(self, args, client_id, graph, data_loader, num_features,
                 num_labels, model):
        self.args = args
        # data
        self.client_id = client_id
        self.graph = graph
        self.data_loader = data_loader
        self.num_features = num_features
        self.num_labels = num_labels
        # process data
        # self.preprocess()
        # share
        self.model = model

        # set seed
        # seed_everything()

        # train and test split
        # self.train_test_split()


    def train_test_split(self):
        idx = np.arange(self.graph.num_nodes)
        random.shuffle(idx)

        train_size = int(self.graph.num_nodes * 0.1)
        val_size = int(self.graph.num_nodes * 0.1)
        test_size = int(self.graph.num_nodes * 0.8)

        train_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

        train_mask[idx[:train_size]] = True
        val_mask[idx[train_size: train_size + val_size]] = True
        test_mask[idx[train_size + val_size:]] = True
        self.graph.train_mask = train_mask
        self.graph.val_mask = val_mask
        self.graph.test_mask = test_mask

        # --------------------logout-------------------
        print("------------------------------------------------client {}----------------------------------------------".format(self.client_id))
        print(self.graph)
        print("node label: ", self.num_labels)
        for label in self.graph.y.unique():
            print("{} label node number: {} | train: {} | val: {} | test: {}".format(label, self.graph.y.eq(label).sum(), 
                                                                                            self.graph.y[train_mask].eq(label).sum(), 
                                                                                            self.graph.y[val_mask].eq(label).sum(),
                                                                                            self.graph.y[test_mask].eq(label).sum()))

    def preprocess(self):

        self.edge_index, self.edge_weight = compute_ppmi(self.graph.edge_index, None, self.graph.num_nodes, self.args.path_len, gpu='cpu')
        
        row, col = self.edge_index
        deg = scatter_add(self.edge_weight, col, dim=0, dim_size=self.graph.num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        row_norm_edge_weight = (self.edge_weight * deg_inv_sqrt[col]).type(torch.float32)
        
        self.x_n = scatter_add(self.graph.x[row] * row_norm_edge_weight.unsqueeze(1), col, dim=0, dim_size=self.graph.num_nodes)
        self.x = self.graph.x
        self.y = self.graph.y

    
    def client_update(self, is_target: bool = False):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        losses = []
        batch_size = self.args.batch_size

        # self.x = self.x.to(self.args.gpu)
        # self.x_n = self.x_n.to(self.args.gpu)
        # self.y = self.y.to(self.args.gpu)
        self.model.to(self.args.gpu)

        self.model.train()
        for i in range(self.args.local_epoch):
            
            # batches = batch_generator([self.x, self.x_n, self.y, self.graph.train_mask], self.args.batch_size, shuffle=True)
            # num_batch = round(self.graph.num_nodes / self.args.batch_size)

            p = float(i) / self.args.local_epoch
            grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
            GradReverse.rate = grl_lambda

            # for _ in range(num_batch):
            for batch in self.data_loader:
                batch.to(self.args.gpu)

                # x_nei_y_batch_mask, shuffle_index = next(batches)
                # x_batch, x_n_batch, y_batch, train_mask = x_nei_y_batch_mask
                
                if is_target:
                    domain_label = torch.ones(batch_size).long().to(self.args.gpu)
                else:
                    domain_label = torch.zeros(batch_size).long().to(self.args.gpu)

                # topological proximity matrix between nodes in each mini-batch
                shuffle_index = batch.n_id[:batch_size]
                a = batch_ppmi(self.graph.num_nodes, shuffle_index, self.graph.edge_index, self.graph.edge_weight)
                a = a.to(self.args.gpu)

                # emb, pred_logit, d_logit = self.model(x_batch, x_n_batch)
                emb, pred_logit, d_logit = self.model(batch.x, batch.edge_index, batch.edge_weight)
                
                total_loss = 0.
                if not is_target:
                    cls_loss = F.cross_entropy(pred_logit[:batch_size][batch.train_mask[:batch_size]], batch.y[:batch_size][batch.train_mask[:batch_size]])
                    total_loss += cls_loss

                total_loss += self.args.domain_coef * F.cross_entropy(d_logit[:batch_size], domain_label)

                net_pro_loss = NetworkEmbedding.net_pro_loss(emb[:batch_size], a)
                total_loss += self.args.net_pro_w * net_pro_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                losses.append(total_loss.item())

        if is_target:
            state = param_filter(self.model.state_dict(), ['node_classifier','domain_discriminator'])
        else:
            state = param_filter(self.model.state_dict(), ['domain_discriminator'])

        paraminfo = dict(
            n_samples = self.graph.num_nodes,
            params = serialize(state),
        )

        return np.mean(losses), paraminfo, serialize(self.model.domain_discriminator.state_dict())

    
    def dm_update(self, dm_param):
        self.model.domain_discriminator.load_state_dict(deserialize(dm_param), strict=False)


    def client_eval(self, istest=False):
        
        self.model.to(self.args.gpu)
        self.graph.to(self.args.gpu)
        # self.x = self.x.to(self.args.gpu)
        # self.x_n = self.x_n.to(self.args.gpu)
        # self.y = self.y.to(self.args.gpu)
        
        results = dict()
        
        self.model.eval()
        with torch.no_grad():
            GradReverse.rate = 1.0

            _, logits, _ = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_weight)
            preds = logits.softmax(dim=-1)
            targets = self.graph.y

            if not istest: 
                results['micro_f1'] = f1_score(preds=preds[self.graph.val_mask], target=targets[self.graph.val_mask], average='micro').item()
                results['macro_f1'] = f1_score(preds=preds[self.graph.val_mask], target=targets[self.graph.val_mask], average='macro', num_classes=self.num_labels).item()
            else:
                results['micro_f1'] = f1_score(preds=preds[self.graph.test_mask], target=targets[self.graph.test_mask], average='micro').item()
                results['macro_f1'] = f1_score(preds=preds[self.graph.test_mask], target=targets[self.graph.test_mask], average='macro', num_classes=self.num_labels).item()

        return results


class ACDNE(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.writer = SummaryWriter(os.path.join(self.args.tb_log_dir, self.args.exp_name))
        self.logger = Log(self.args.log_dir, self.args.exp_name).get_logger()
        self.logger.info(json.dumps(vars(args)))

        # state dir
        self.state_path = os.path.join(self.args.state_dir, self.args.exp_name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # load data
        self.subgraphs, self.num_graphs, self.num_features, self.num_labels = get_data(self.args.data_path)

        # clients
        self.num_clients = self.num_graphs
        self.clients = []
        for i in range(self.num_clients):
            model = ACDNE_model(
                n_input=self.num_features,
                n_hidden=[self.args.h_dim] * self.args.num_layers,
                n_emb=self.args.h_dim,
                num_class=self.num_labels,
                dropout=self.args.dropout
            )

            if i == self.num_clients - 1:
                self.subgraphs[i].test_mask[:] = True
            
            idx = torch.arange(self.subgraphs[i].num_nodes)
            self.subgraphs[i].n_id = idx
            self.subgraphs[i].edge_index, self.subgraphs[i].edge_weight = compute_ppmi(self.subgraphs[i].edge_index, None, self.subgraphs[i].num_nodes, self.args.path_len, gpu='cpu')
            data_loader = NeighborLoader(self.subgraphs[i], input_nodes=idx, num_neighbors=[-1]*self.args.num_layers,
                                            shuffle=True, batch_size=self.args.batch_size)

            client = Client(self.args, i, self.subgraphs[i], data_loader, self.num_features, self.num_labels, model)
            self.clients.append(client)

        # server
        self.server = Server(self.args, 
            model=ACDNE_model(
                n_input=self.num_features,
                n_hidden=[self.args.h_dim] * self.args.num_layers,
                n_emb=self.args.h_dim,
                num_class=self.num_labels,
                dropout=self.args.dropout
            )
        )

    def write_training_loss(self, loss, e):
        self.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/micro_f1", results['micro_f1'], e)
        self.writer.add_scalar("evaluation/macro_f1", results['macro_f1'], e)

    def save_checkpoint(self, e):
        state = self.server.model.state_dict()
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.args.exp_name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.state_path,
                                    self.args.exp_name + '.' + str(e) + '.ckpt'))
    

    def save_model(self, epoch):
        os.rename(os.path.join(self.state_path, self.args.exp_name + '.' + str(epoch) + '.ckpt'),
                  os.path.join(self.state_path, self.args.exp_name + '.best'))


    def send_global_model(self):
        for k, client in enumerate(self.clients):
            client.model.load_state_dict(deserialize(self.server.send_global_model()), strict=False)


    def train(self):
        best_epoch = 0
        best_micro_f1 = 0
        best_macro_f1 = 0

        for num_round in range(1, self.args.max_round + 1):
            
            self.send_global_model()

            round_loss = 0
            for client_id in range(self.num_clients-1):
                source_client_loss, source_paraminfo, source_dm_param = self.clients[client_id].client_update(is_target=False)
                self.clients[self.num_clients - 1].dm_update(source_dm_param)
                target_client_loss, target_paraminfo, target_dm_param = self.clients[self.num_clients - 1].client_update(is_target=True)
                self.clients[client_id].dm_update(target_dm_param)

                round_loss += (source_client_loss + target_client_loss) / 2
                
                paraminfo = self.server.federated_average([source_paraminfo, target_paraminfo])
                self.server.receive_parameters_info(paraminfo=paraminfo)
            
            round_loss /= (self.num_clients-1)

            self.server.aggregation()

            self.logger.info('round: {} | loss: {:.4f}'.format(num_round, round_loss))
            self.write_training_loss(round_loss, num_round)

            if num_round % self.args.check_per_round == 0:
                eval_res = self.evaluate(istest=True)
                self.write_evaluation_result(eval_res, num_round)

                if eval_res['micro_f1'] + eval_res['macro_f1'] > best_micro_f1 + best_macro_f1:
                    best_micro_f1 = eval_res['micro_f1']
                    best_macro_f1 = eval_res['macro_f1']
                    best_epoch = num_round
                    self.logger.info('best target model | micro_f1: {:.4f}, macro_f1: {:.4f}'.format(best_micro_f1, best_macro_f1))
                    self.save_checkpoint(num_round)
                else:
                    self.logger.info('best target model is at round {0}, micro_f1: {1:.4f}, macro_f1: {2:.4f}'.format(
                        best_epoch, best_micro_f1, best_macro_f1))

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_epoch)
        self.before_test_load()
        self.evaluate(istest=True)

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.args.exp_name + '.best'), map_location=self.args.gpu)
        self.server.model.load_state_dict(state)

    def evaluate(self, istest=False):
        self.send_global_model()
        result = dict()
        avg_micro_f1 = 0
        avg_macro_f1 = 0

        for idx, client in enumerate(self.clients):
            client_res = client.client_eval(istest)

            self.logger.info('cli_id: {} | micro_f1: {:.4f}, macro_f1: {:.4f}'.format(
                    idx,
                    client_res['micro_f1'], client_res['macro_f1']
                ))

            if idx == self.num_clients - 1 :
                result['micro_f1'] = client_res['micro_f1']
                result['macro_f1'] = client_res['macro_f1']

            avg_micro_f1 += client_res['micro_f1']
            avg_macro_f1 += client_res['macro_f1']

        avg_micro_f1 /= self.num_clients
        avg_macro_f1 /= self.num_clients

        self.logger.info('cli_avg | micro_f1: {:.4f}, macro_f1: {:.4f}'.format(
                    avg_micro_f1, avg_macro_f1
                ))

        return result
