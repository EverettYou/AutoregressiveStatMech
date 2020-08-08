import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

from phys import *
from auto import *
from flow import *


class HolographicPixelFlow(nn.Module, dist.TransformedDistribution):
    """ Combination of hierarchical autoregressive and flow-based model for lattice models.
    
        Args:
        model: a energy model to learn
        hidden_features: a list of feature dimensions of hidden layers
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    def __init__(self, model: EnergyModel, hidden_features, nonlinearity: str = 'ReLU', bias: bool = True):
        super(HolographicPixelFlow, self).__init__()
        self.model = model
        self.haar = HaarTransform(model.lattice)
        n = model.lattice.group.order
        self.onecat = OneHotCategoricalTransform(n)
        features = [n] + hidden_features + [n]
        auto = AutoregressiveModel(model.lattice.units, features, nonlinearity, bias)
        dist.TransformedDistribution.__init__(self, auto, [self.onecat, self.haar])
        self.transform = dist.ComposeTransform(self.transforms)
        
    def energy(self, input): # create a shortcut for energy
        return self.model.energy(input)