import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

from phys import *
from auto import *
from flow import *


class HpGCN(nn.Module, dist.TransformedDistribution):
    """ Combination of hierarchical autoregressive and flow-based model for lattice models.
    
        Args:
        energy: a energy model to learn
        hidden_features: a list of feature dimensions of hidden layers
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    def __init__(self, energy: EnergyModel, edge_features: int, hidden_node_features,
                nonlinearity: str = 'ReLU', bias: bool = True):
        super(HpGCN, self).__init__()
        self.energy = energy
        self.group = energy.group
        self.lattice = energy.lattice
        self.haar = HaarTransform(self.group, self.lattice)
        self.onecat = OneHotCategoricalTransform(self.group.order)
        node_features = [self.group.order] + hidden_node_features + [self.group.order]
        auto = AutoregressiveModel(self.lattice, edge_features, node_features, nonlinearity, bias)
        dist.TransformedDistribution.__init__(self, auto, [self.onecat, self.haar])
        self.transform = dist.ComposeTransform(self.transforms)