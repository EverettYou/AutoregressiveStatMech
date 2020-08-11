import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math
from torch_geometric.nn import MessagePassing
class GraphConv(MessagePassing):
    """ Graph Convolution layer 
        
        Args:
        in_features: number of input features per node
        out_features: number of output features per node 
        edge_features: number of features per edge
    """
    def __init__(self, in_features: int, out_features: int, edge_features: int, bias: bool = True):
        super(GraphConv, self).__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.weight = nn.Parameter(torch.Tensor(edge_features, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(edge_features, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, edge_features={}, bias={}'.format(
            self.in_features, self.out_features, self.edge_features, self.bias is not None)

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x, edge_index, edge_attr):
        # x: shape [..., N, in_features]
        # edge_index: shape [2, E]
        # edge_attr: shape [E, edge_features]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: shape [..., E, in_features]
        # edge_attr: [E, edge_features]
        weight = torch.tensordot(edge_attr, self.weight, dims=1)
        x_j = torch.sum(weight * x_j.unsqueeze(-2), -1)
        if self.bias is not None:
            bias = torch.tensordot(edge_attr, self.bias, dims=1)
            x_j += bias
        return x_j
    
    def forward_from(self, x, i, edge_index, edge_attr):
        mask = (edge_index[0] == i)
        return self(x, edge_index[:, mask], edge_attr[mask, :])
    

from phys import LatticeSystem
class AutoregressiveModel(nn.Module, dist.Distribution):
    """ Represent a generative model that can generate samples and evaluate log probabilities.
        
        Args:
        nodes: number of units in the model
        features: a list of feature dimensions from the input layer to the output layer
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    
    def __init__(self, lattice: LatticeSystem, edge_features: int, node_features, 
                 nonlinearity: str = 'ReLU', bias: bool = True):
        super(AutoregressiveModel, self).__init__()
        self.lattice = lattice
        self.nodes = self.lattice.sites
        self.edge_index = self.lattice.edge_index
        self.edge_type = self.lattice.edge_type
        self.edge_index_ext, self.edge_type_ext = self.edge_extension()
        self.num_edge_type = self.edge_type.max() + 1
        self.edge_features = edge_features
        self.edge_embedding = nn.Sequential(
            nn.Embedding(self.num_edge_type, self.edge_features),
            nn.Softmax(-1)
            )
        if isinstance(node_features, int):
            self.node_features = [node_features, node_features]
        else:
            if node_features[0] != node_features[-1]:
                raise ValueError('In features {}, the first and last feature dimensions must be equal.'.format(features))
            self.node_features = node_features
        self.layers = nn.ModuleList()
        for l in range(1, len(self.node_features)):
            if l > 1: 
                self.layers.append(getattr(nn, nonlinearity)())
            self.layers.append(GraphConv(self.node_features[l - 1], self.node_features[l], self.edge_features, bias))
        dist.Distribution.__init__(self, event_shape=torch.Size([self.nodes, self.node_features[0]]))
        self.has_rsample = True
    
    def edge_extension(self):
        node_list = torch.arange(1, self.nodes)
        edge_index = torch.stack((node_list, node_list), 0)
        edge_type = torch.zeros(edge_index.size(-1), dtype=self.edge_type.dtype)
        edge_index_ext = torch.cat((edge_index, self.edge_index), -1)
        edge_type_ext = torch.cat((edge_type, self.edge_type), -1)
        return edge_index_ext, edge_type_ext
    
    
    def extra_repr(self):
        return '(nodes): {}, (edge_features): {}, (node_features): {}'.format(self.nodes, self.edge_features, self.node_features) + super(AutoregressiveModel, self).extra_repr()
          
    def forward(self, input):
        edge_attr = self.edge_embedding(self.edge_type)
        edge_attr_ext = self.edge_embedding(self.edge_type_ext)
        for l, layer in enumerate(self.layers): # apply layers
            if isinstance(layer, GraphConv): # for graph convolution layers
                if l == 0: # first layer
                    output = layer(input, self.edge_index, edge_attr)
                else: # remaining layers
                    output = layer(output, self.edge_index_ext, edge_attr_ext)
            else: # activation layers
                output = layer(output)
        return output # logits
    
    def log_prob(self, value):
        logits = self(value) # forward pass to get logits
        return torch.sum(value * F.log_softmax(logits, dim=-1), (-2,-1))

    def sampler(self, logits, dim=-1): # simplified from F.gumbel_softmax
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels += logits
        index = gumbels.max(dim, keepdim=True)[1]
        return torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)

    def _sample(self, batch_size: int, sampler = None):
        if sampler is None: # use default sampler
            sampler = self.sampler
        # create a list of tensors to cache layer-wise outputs
        cache = [torch.zeros(batch_size, self.nodes, self.node_features[0])]
        for l, layer in enumerate(self.layers):
            if isinstance(layer, GraphConv): # for graph convolution layers
                node_features = layer.out_features
                cache.append(torch.zeros(batch_size, self.nodes, node_features))
            else: # activation layers
                cache.append(torch.zeros(batch_size, self.nodes, node_features))
        # autoregressive batch sampling
        edge_attr = self.edge_embedding(self.edge_type)
        edge_attr_ext = self.edge_embedding(self.edge_type_ext)
        cache[0][..., 0, :] = sampler(cache[0][..., 0, :]) # always sample node 0 uniformly
        for i in range(1, self.nodes):
            for l, layer in enumerate(self.layers):
                if isinstance(layer, GraphConv): # for graph convolution layers
                    if l==0: # first layer
                        cache[l + 1] += layer.forward_from(cache[l], i - 1, self.edge_index, edge_attr)
                    else: # remaining layers
                        cache[l + 1] += layer.forward_from(cache[l], i, self.edge_index_ext, edge_attr_ext)
                else: # activation layers
                    cache[l + 1][..., i, :] = layer(cache[l][..., i, :])
            # the last cache hosts the logit, sample from it 
            cache[0][..., i, :] = sampler(cache[-1][..., i, :])
        return cache # cache[0] hosts the sample
    
    def sample(self, batch_size=1):
        with torch.no_grad():
            cache = self._sample(batch_size)
        return cache[0]
    
    def rsample(self, batch_size=1, tau=None, hard=False):
        if tau is None: # if temperature not given
            tau = 1/(self.features[-1]-1) # set by the out feature dimension
        cache = self._sample(batch_size, lambda x: F.gumbel_softmax(x, tau, hard))
        return cache[0]