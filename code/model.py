import torch
import torch.nn as nn
import torch.nn.functional as F
from container import Sequential
from typing import Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import GCNConv, SAGEConv
from utils import *
from torch.distributions.laplace import Laplace


act_dict = dict(relu="ReLU",
                relu6="ReLU6",
                sigmoid="Sigmoid",
                celu="CELU",
                elu="ELU",
                gelu="GELU",
                leakyrelu="LeakyReLU",
                prelu="PReLU",
                selu="SELU",
                silu="SiLU",
                softmax="Softmax",
                tanh="Tanh")


def activations(act: Optional[str] = None,
                inplace: bool = False) -> torch.nn.Module:

    out = act_dict.get(act, None)
    if out:
        return getattr(nn, out)()
    else:
        raise ValueError(f"Unknown activation {act}. The allowed activation functions are {tuple(act_dict.keys())}.")


class PPMIConv(GCNConv):
    def __init__(self, in_channels: int, out_channels: int, weight_data=None, bias_data=None,
                        improved: bool = False, cached: bool = False, add_self_loops: bool = True, 
                        normalize: bool = True, bias: bool = True, path_len=5, gpu="cuda:0", **kwargs):
        super().__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias, **kwargs)
        
        if weight_data is not None:
            self.lin.weight = weight_data
            print("use shared weight")

        if bias_data is not None:
            self.bias = bias_data
            print("use shared bias")

        self.cached_edge_index = {}

        self.path_len = path_len

        self.gpu = gpu

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, cached_name = None) -> Tensor:

        assert(cached_name != None)

        if self.normalize:
            cache = self.cached_edge_index.setdefault(cached_name, None)
            if cache is None:
                edge_index, edge_weight = ppmi_norm(
                    edge_index, edge_weight, x.size(self.node_dim), self.path_len,
                    self.improved, self.add_self_loops, gpu=self.gpu)
                if self.cached:
                    self.cached_edge_index[cached_name] = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits


class Projection(nn.Module):
    def __init__(self, n_h):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(n_h, n_h)
        self.fc2 = nn.Linear(n_h, n_h)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 dropout=0.5,
                 bias=True,
                 normalize=True,
                 base_model=None,
                 type="gcn",
                 **kwargs):

        super(GCN, self).__init__()

        if base_model is None:
            weights = [None] * (len(hids) + 1)
            biases = [None] * (len(hids) + 1)
        else:
            weights = []
            biases = []
            for conv_layer in base_model.conv:
                if isinstance(conv_layer, GCNConv):
                    weights.append(conv_layer.lin.weight)
                    biases.append(conv_layer.bias)

        conv = nn.ModuleList()
        act_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.type = type
        for hid, act, weight_data, bias_data in zip(hids, acts, weights[:-1], biases[:-1]):
            if type == 'gcn':
                conv.append(GCNConv(in_features,
                                    hid,
                                    cached=False,
                                    bias=bias,
                                    normalize=normalize,
                                    **kwargs))
            elif type == 'ppmi':
                conv.append(PPMIConv(in_features,
                                    hid,
                                    weight_data=weight_data,
                                    bias_data=bias_data,
                                    cached=True,
                                    bias=bias,
                                    normalize=normalize,
                                    **kwargs))

            act_layers.append(activations(act=act))
            in_features = hid

        if type == 'gcn':
            conv.append(GCNConv(in_features,
                                out_features,
                                cached=False,
                                bias=bias,
                                normalize=normalize,
                                **kwargs))
        elif type == 'ppmi':
            conv.append(PPMIConv(in_features,
                                out_features,
                                weight_data=weights[-1],
                                bias_data=biases[-1],
                                cached=True,
                                bias=bias,
                                normalize=normalize,
                                **kwargs))
        self.conv = conv
        self.acts = act_layers


    def forward(self, x, edge_index, cached_name='default'):
        x = self.dropout(x)
        for i, conv in enumerate(self.conv):
            if self.type == 'gcn':
                x = conv(x, edge_index)
            elif self.type == 'ppmi':
                x = conv(x, edge_index, cached_name=cached_name)
            if i != len(self.conv) - 1:
                x = self.acts[i](x)
                x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hids=[16],
                 acts=['relu'],
                 bn=True,
                 dropout=0.5,
                 bias=True,
                 **kwargs):

        super().__init__()

        lin = []
        lin.append(nn.Dropout(dropout))

        for hid, act in zip(hids, acts):
            if bn:
                lin.append(nn.BatchNorm1d(in_features))
            lin.append(nn.Linear(in_features,
                                hid,
                                bias=bias))
            lin.append(activations(act=act))
            lin.append(nn.Dropout(dropout))
            in_features = hid
        if bn:
            lin.append(nn.BatchNorm1d(in_features))
        lin.append(nn.Linear(in_features,
                            out_features,
                            bias=bias))
        lin = Sequential(*lin)

        self.lin = lin

    def forward(self, x):
        return self.lin(x)


class FE1(nn.Module):
    def __init__(self, n_input, n_hidden, dropout) -> None:
        super(FE1, self).__init__()
        self.dropout = dropout
        self.h1_self = nn.Linear(n_input, n_hidden[0])
        self.h2_self = nn.Linear(n_hidden[0], n_hidden[1])

        # init
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_self.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h1_self.weight, 1.414)
        nn.init.constant_(self.h1_self.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_self.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h2_self.weight, 1.414)
        nn.init.constant_(self.h2_self.bias, 0.1)

    def forward(self, x):
        x = F.dropout(F.relu(self.h1_self(x)), self.dropout)
        return F.relu(self.h2_self(x))


class FE2(nn.Module):
    def __init__(self, n_input, n_hidden, dropout):
        super(FE2, self).__init__()
        self.dropout = dropout
        self.h1_nei = nn.Linear(n_input, n_hidden[0])
        self.h2_nei = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_nei.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h1_nei.weight, 1.414)
        nn.init.constant_(self.h1_nei.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_nei.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h2_nei.weight, 1.414)
        nn.init.constant_(self.h2_nei.bias, 0.1)

    def forward(self, x_nei):
        x_nei = F.dropout(F.relu(self.h1_nei(x_nei)), self.dropout)
        return F.relu(self.h2_nei(x_nei))


class NetworkEmbedding(nn.Module):
    def __init__(self, n_input, n_hidden, n_emb, dropout, batch_size):
        super(NetworkEmbedding, self).__init__()
        self.drop = dropout
        self.batch_size = batch_size
        self.fe1 = FE1(n_input, n_hidden, dropout)
        self.fe2 = FE2(n_input, n_hidden, dropout)
        self.emb = nn.Linear(n_hidden[-1]*2, n_emb)
        std = 1/(n_hidden[-1]*2)**0.5
        nn.init.trunc_normal_(self.emb.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.emb.weight, 1.414)
        nn.init.constant_(self.emb.bias, 0.1)

    def forward(self, x, x_nei):
        h2_self = self.fe1(x)
        h2_nei = self.fe2(x_nei)
        return F.relu(self.emb(torch.cat((h2_self, h2_nei), 1)))

    # def pairwise_constraint(self, emb):
    #     emb_s = emb[:int(self.batch_size/2), :]
    #     emb_t = emb[int(self.batch_size/2):, :]
    #     return emb_s, emb_t

    @staticmethod
    def net_pro_loss(emb, a):
        r = torch.sum(emb*emb, 1)
        r = torch.reshape(r, (-1, 1))
        dis = r-2*torch.matmul(emb, emb.T)+r.T
        return torch.mean(torch.sum(a.__mul__(dis), 1))


class NodeClassifier(nn.Module):
    def __init__(self, n_emb, num_class):
        super(NodeClassifier, self).__init__()
        self.layer = nn.Linear(n_emb, num_class)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.layer.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.layer.weight, 1.414)
        nn.init.constant_(self.layer.bias, 0.1)

    def forward(self, emb):
        pred_logit = self.layer(emb)
        return pred_logit


class DomainDiscriminator(nn.Module):
    def __init__(self, n_emb):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(n_emb, 128)
        self.h_dann_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 2)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h_dann_1.weight, 1.414)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        # nn.init.xavier_normal_(self.h_dann_2.weight, 1.414)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        # nn.init.xavier_normal_(self.output_layer.weight, 1.414)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


class ACDNE_model(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_emb,
                 num_class,
                 dropout=0.5):
        super(ACDNE_model, self).__init__()
        # self.network_embedding = NetworkEmbedding(n_input, n_hidden, n_emb, dropout, batch_size)
        self.network_embedding = GCN(n_input, n_hidden[-1], n_hidden[:-1], ['relu']*len(n_hidden[:-1]), dropout)
        self.node_classifier = NodeClassifier(n_emb, num_class)
        self.domain_discriminator = DomainDiscriminator(n_emb)
        self.grl = GRL()

    def forward(self, x, edge_index, edge_weight):
        # 2 layers Network_Embedding
        emb = self.network_embedding(x, edge_index, edge_weight)
        # 1 layer Node_Classifier
        pred_logit = self.node_classifier(emb)
        # Domain_Discriminator
        h_grl = self.grl(emb)
        d_logit = self.domain_discriminator(h_grl)
        
        return emb, pred_logit, d_logit


class Teacher:
    """Implementation of teacher models.
       Teacher models are ensemble of models which learns directly disjoint splits of the sensitive data
       The ensemble of teachers are further used to label unlabelled public data on which the student is 
       trained. 
       Args:
           args[Arguments object]: An object of Arguments class with required hyperparameters
           n_teachers[int]: Number of teachers
           epochs[int]: Number of epochs to train each model
    """

    def __init__(self, model, n_teachers=1, n_emb=256, epsilon=0.5):

        self.n_teachers = n_teachers
        self.n_emb = n_emb
        self.model = model
        self.models = {}
        self.init_models()
        self.epsilon = epsilon

    def init_models(self):
        """Initialize teacher models according to number of required teachers"""

        name = "model_"
        for index in range(0, self.n_teachers):

            model = self.model(self.n_emb)
            self.models[name + str(index)] = model

    def addnoise(self, x):
        """Adds Laplacian noise to histogram of counts
           Args:
                counts[torch tensor]: Histogram counts
                epsilon[integer]:Amount of Noise
           Returns:
                counts[torch tensor]: Noisy histogram of counts
        """

        m = Laplace(torch.tensor([0.0]), torch.tensor([self.epsilon]))
        count = x + m.sample()

        return count

    def split(self, dataset):
        """Function to split the dataset into non-overlapping subsets of the data
           Args:
               dataset[torch tensor]: The dataset in the form of (image,label)
           Returns:
               split: Split of dataset
        """

        ratio = int(len(dataset) / self.n_teachers)

        split = torch.split(dataset, [ratio] * self.n_teachers, dim=0)

        return split

    def forward(self, dataset):
        """Function to train all teacher models.
           Args:
                dataset[torch tensor]: Dataset used to train teachers in format (embedding)
        """

        all_output = []
        split = self.split(dataset)

        for index, model_name in enumerate(self.models):
            
            self.models[model_name].to(split[index].device)
            output = self.models[model_name](split[index])
            all_output.append(output)
        
        return torch.vstack(all_output)


    def aggregate(self, model_votes, batch_size):
        """Aggregate model output into a single tensor of votes of all models.
           Args:
                votes: Model output
                n_dataset: Number of datapoints
           Returns:
                counts: Torch tensor with counts across all models    
           """

        counts = torch.zeros([batch_size, 2])
        model_counts = torch.zeros([self.n_teachers, batch_size])
        model_index = 0

        for model in model_votes:

            index = 0

            for tensor in model_votes[model]:
                for val in tensor:

                    counts[index][val] += 1
                    model_counts[model_index][index] = val
                    index += 1

            model_index += 1

        return counts, model_counts


    def predict(self, data):
        """Make predictions using Noisy-max using Laplace mechanism.
           Args:
                data: Data for which predictions are to be made
           Returns:
                predictions: Predictions for the data
        """

        model_predictions = {}

        for model in self.models:

            out = []
            output = self.models[model](data)
            output = output.max(dim=1)[1]
            out.append(output)

            model_predictions[model] = out

        counts, model_counts = self.aggregate(model_predictions, len(data))
        counts = counts.apply_(self.addnoise)

        predictions = []

        for batch in counts:

            predictions.append(torch.tensor(batch.max(dim=0)[1].long()).clone().detach())

        output = {"predictions": predictions, "counts": counts, "model_counts": model_counts}

        return output


class FGNN_model(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_emb,
                 num_class,
                 dropout=0.5,
                 batch_size=128):
        super(FGNN_model, self).__init__()
        self.batch_size = batch_size
        # self.network_embedding = NetworkEmbedding(n_input, n_hidden, n_emb, dropout, batch_size)
        self.network_embedding = GCN(n_input, n_hidden[-1], n_hidden[:-1], ['relu']*len(n_hidden[:-1]), dropout)
        self.node_classifier = NodeClassifier(n_emb, num_class)
        # self.domain_discriminator = DomainDiscriminator(n_emb)
        self.domain_discriminators = Teacher(model=DomainDiscriminator, n_teachers=8, n_emb=n_emb)
        self.domain_student = DomainDiscriminator(n_emb)
        self.grl = GRL()

    def forward(self, x, edge_index, edge_weight):
        # 2 layers Network_Embedding
        emb = self.network_embedding(x, edge_index, edge_weight)
        # 1 layer Node_Classifier
        pred_logit = self.node_classifier(emb)
        # Domain_Discriminator
        h_grl = self.grl(emb)
        d_logits = self.domain_discriminators.forward(h_grl.detach()[:self.batch_size])
        stu_logits = self.domain_student(h_grl)
        
        return emb, pred_logit, d_logits, stu_logits


model_dict = dict(
    gcn=GCN,
    mlp=MLP
)


def models(model_type: Optional[str] = None, 
           in_features = None, 
           out_features = None, 
           layer_num: Optional[int] = 1, 
           h_dim: Optional[int] = 16,
           act: Optional[str] = 'relu',
           **kwargs) -> torch.nn.Module:

    hids = [h_dim] * (layer_num - 1)
    acts = [act] * (layer_num - 1)

    model = model_dict.get(model_type, None)
    if model:
        return model(in_features, out_features, hids=hids, acts=acts, **kwargs)
    else:
        raise ValueError(f"Unknown model {model_type}. The allowed models are {tuple(model_type.keys())}.")
