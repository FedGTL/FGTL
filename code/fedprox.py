import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import models
from torchmetrics.functional import accuracy, f1_score


class Server(object):
    def __init__(self, args, encoder: nn.Module, classifier: nn.Module):
        self.args = args
        self.encoder = encoder
        self.classifier = classifier
        self.current_round_parames = []

    def send_global_encoder(self):
        return copy.deepcopy(serialize(self.encoder.state_dict()))

    def send_global_classifier(self):
        return copy.deepcopy(serialize(self.classifier.state_dict()))

    def receive_parameters_info(self, paraminfo):
        self.current_round_parames.append(paraminfo)

    def aggregation(self):

        total_encoder_params = {}
        total_classifier_params = {}
        n_total_samples = 0
        for paraminfo in self.current_round_parames:
            n_samples = paraminfo['n_samples']
            
            if 'encoder_params' in paraminfo.keys():
                for k, v in deserialize(paraminfo['encoder_params']).items():
                    if k not in total_encoder_params:
                        total_encoder_params[k] = []
                    total_encoder_params[k].append(v.cpu().numpy() * n_samples)
            
            if 'classifier_params' in paraminfo.keys():
                for k, v in deserialize(paraminfo['classifier_params']).items():
                    if k not in total_classifier_params:
                        total_classifier_params[k] = []
                    total_classifier_params[k].append(v.cpu().numpy() * n_samples)

            n_total_samples += n_samples

        if len(total_encoder_params) != 0:
            encoder_parameters = {}
            for k, v in total_encoder_params.items():
                encoder_parameters[k] = torch.tensor(v).sum(dim=0) / n_total_samples
            self.encoder.load_state_dict(encoder_parameters)
        
        if len(total_classifier_params) != 0:
            classifier_parameters = {}
            for k, v in total_classifier_params.items():
                classifier_parameters[k] = torch.tensor(v).sum(dim=0) / n_total_samples
            self.classifier.load_state_dict(classifier_parameters)
        
        self.current_round_parames.clear()


class Client(object):
    def __init__(self, args, client_id, graph, num_features,
                 num_labels, encoder, classifier):
        self.args = args
        # data
        self.client_id = client_id
        self.graph = graph
        self.num_features = num_features
        self.num_labels = num_labels
        # share
        self.encoder = encoder
        self.classifier = classifier

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

    def client_update(self):

        optimizer = optim.Adam([
                        {'params': self.encoder.parameters()}, 
                        {'params': self.classifier.parameters()}
                        ], lr=self.args.lr, weight_decay=self.args.weight_decay)

        losses = []

        self.graph.to(self.args.gpu)
        self.encoder.to(self.args.gpu)
        self.classifier.to(self.args.gpu)

        global_encoder = copy.deepcopy(self.encoder)
        global_encoder_weight_collector = list(global_encoder.parameters())
        global_classifier = copy.deepcopy(self.classifier)
        global_classifier_weight_collector = list(global_classifier.parameters())

        self.encoder.train()
        self.classifier.train()
        for i in range(self.args.local_epoch):
            embeddings = self.encoder(self.graph.x, self.graph.edge_index)
            logits = self.classifier(embeddings)
            loss = F.cross_entropy(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask])

            # for fedprox
            # mu = 0.01 or 0.001
            fed_prox_reg = 0.0
            for param_index, param in enumerate(self.encoder.parameters()):
                fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_encoder_weight_collector[param_index]))**2)
            for param_index, param in enumerate(self.classifier.parameters()):
                fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_classifier_weight_collector[param_index]))**2)

            loss += fed_prox_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        paraminfo = dict(
            n_samples = self.graph.num_nodes,
            encoder_params = serialize(self.encoder.state_dict()),
            classifier_params = serialize(self.classifier.state_dict())
        )

        return np.mean(losses), paraminfo

    def client_eval(self, istest=False):
        
        self.encoder.to(self.args.gpu)
        self.classifier.to(self.args.gpu)
        self.graph.to(self.args.gpu)

        results = dict()
        
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            embeds = self.encoder(self.graph.x, self.graph.edge_index)
            logits = self.classifier(embeds)
            preds = logits.softmax(dim=-1)
            targets = self.graph.y

            if not istest: 
                results['micro_f1'] = f1_score(preds=preds[self.graph.val_mask], target=targets[self.graph.val_mask], average='micro').item()
                results['macro_f1'] = f1_score(preds=preds[self.graph.val_mask], target=targets[self.graph.val_mask], average='macro', num_classes=self.num_labels).item()
            else:
                results['micro_f1'] = f1_score(preds=preds[self.graph.test_mask], target=targets[self.graph.test_mask], average='micro').item()
                results['macro_f1'] = f1_score(preds=preds[self.graph.test_mask], target=targets[self.graph.test_mask], average='macro', num_classes=self.num_labels).item()

        return results


class FedProx(object):
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
            encoder = models(
                            model_type=self.args.cli_model, 
                            in_features=self.num_features,
                            out_features=self.args.h_dim,
                            layer_num=self.args.num_layers,
                            h_dim=self.args.h_dim,
                            act=self.args.act
            )
            classifier = models(
                model_type='mlp', 
                in_features=self.args.h_dim,
                out_features=self.num_labels,
                layer_num=1,
                h_dim=self.args.h_dim,
                act=self.args.act,
                dropout=self.args.dropout
            )

            if i == self.num_clients - 1:
                self.subgraphs[i].test_mask[:] = True 

            client = Client(self.args, i, self.subgraphs[i], self.num_features, self.num_labels, encoder, classifier)
            self.clients.append(client)

        # server
        self.server = Server(self.args, 
            encoder=models(
                model_type=self.args.ser_model,
                in_features=self.num_features,
                out_features=self.args.h_dim,
                layer_num=self.args.num_layers,
                h_dim=self.args.h_dim,
                act=self.args.act
            ),
            classifier=models(
                model_type='mlp',
                in_features=self.args.h_dim,
                out_features=self.num_labels,
                layer_num=1,
                h_dim=self.args.h_dim,
                act=self.args.act,
                dropout=self.args.dropout
            )
        )

    def write_training_loss(self, loss, e):
        self.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/micro_f1", results['micro_f1'], e)
        self.writer.add_scalar("evaluation/macro_f1", results['macro_f1'], e)

    def save_checkpoint_for_encoder(self, e):
        encoder_state = self.server.encoder.state_dict()
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.args.exp_name+'_encoder' in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save current checkpoint
        torch.save(encoder_state, os.path.join(self.state_path,
                                    self.args.exp_name + '_encoder.' + str(e) + '.ckpt'))
    
    def save_checkpoint_for_classifier(self, e):
        classifier_state = self.server.classifier.state_dict()
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.args.exp_name+'_classifier' in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save current checkpoint
        torch.save(classifier_state, os.path.join(self.state_path,
                                    self.args.exp_name + '_classifier.' + str(e) + '.ckpt'))

    def save_model(self, best_encoder_epoch, best_cls_epoch):
        os.rename(os.path.join(self.state_path, self.args.exp_name + '_encoder.' + str(best_encoder_epoch) + '.ckpt'),
                  os.path.join(self.state_path, self.args.exp_name + '_encoder.best'))
        os.rename(os.path.join(self.state_path, self.args.exp_name + '_classifier.' + str(best_cls_epoch) + '.ckpt'),
                  os.path.join(self.state_path, self.args.exp_name + '_classifier.best'))

    def send_global_model(self, encoder: bool, classifier: bool):
        for k, client in enumerate(self.clients):
            if encoder:
                client.encoder.load_state_dict(deserialize(self.server.send_global_encoder()))
            if classifier:
                client.classifier.load_state_dict(deserialize(self.server.send_global_classifier()))

    def train(self):
        best_epoch = 0
        best_micro_f1 = 0
        best_macro_f1 = 0

        for num_round in range(1, self.args.max_round + 1):
            
            self.send_global_model(encoder=True, classifier=True)

            round_loss = 0
            for client_id in range(self.num_clients-1):
                client_loss, paraminfo = self.clients[client_id].client_update()
                round_loss += client_loss
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
                    self.save_checkpoint_for_encoder(num_round)
                    self.save_checkpoint_for_classifier(num_round)
                else:
                    self.logger.info('best target model is at round {0}, micro_f1: {1:.4f}, macro_f1: {2:.4f}'.format(
                        best_epoch, best_micro_f1, best_macro_f1))

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_epoch, best_epoch)
        self.before_test_load()
        self.evaluate(istest=True)

    def before_test_load(self):
        encoder_state = torch.load(os.path.join(self.state_path, self.args.exp_name + '_encoder.best'), map_location=self.args.gpu)
        self.server.encoder.load_state_dict(encoder_state)
        classifier_state = torch.load(os.path.join(self.state_path, self.args.exp_name + '_classifier.best'), map_location=self.args.gpu)
        self.server.classifier.load_state_dict(classifier_state)

    def evaluate(self, istest=False):
        self.send_global_model(encoder=True, classifier=True)
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
