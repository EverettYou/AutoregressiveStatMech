import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class OneHotCategoricalTransform(dist.Transform):
    """Convert between one-hot and categorical representations.
    
    Args:
    num_classes: number of classes."""
    def __init__(self, num_classes: int):
        super(OneHotCategoricalTransform, self).__init__()
        self.num_classes = num_classes
        self.bijective = True
    
    def _call(self, x):
        # one-hot to categorical
        return x.max(dim=-1)[1]
    
    def _inverse(self, y):
        # categorical to one-hot
        return F.one_hot(y, self.num_classes).to(dtype=torch.float)
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)


def coordinate_select(input: torch.Tensor, coordinate, dims = None):
    """select element form input by coordinate index, coordinate can be any iterable
    (to be called in class HaarTransform)
    
    Args:
    input: input tensor to select from
    coordinate: iterable of indices
    dims: axises of corresponding indices, default: 0,1,...,len(coordintate)
    """
    output = input
    if dims is None:
        dims = range(len(coordinate))
    for dim, i in zip(dims, coordinate):
        output = output.narrow(dim, i, 1)
    return output


import itertools
from phys import LatticeSystem
class HaarTransform(dist.Transform):
    """Haar wavelet transformation (bijective)
    transformation takes real space configurations x to wavelet space encoding y
    
    Args:
    lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, lattice: LatticeSystem):
        super(HaarTransform, self).__init__()
        self.lattice = lattice
        self.bijective = True
        self.make_wavelet()
        self.make_plan()
        
    # construct Haar wavelet basis
    def make_wavelet(self):
        self.wavelet = torch.zeros(torch.Size([self.lattice.units]+self.lattice.shape), dtype=torch.int)
        def partition(rng: torch.Tensor, dim: int, ind: int):
            if rng[dim].sum()%2 == 0:
                mid = rng[dim].sum()//2
                rng1 = rng.clone()
                rng1[dim, 1] = mid
                rng2 = rng.clone()
                rng2[dim, 0] = mid
                wave = self.wavelet[ind]
                for k in range(rng1.size(0)):
                    wave = wave.narrow(k, rng1[k,0], rng1[k,1]-rng1[k,0])
                wave.fill_(1)
                partition(rng1, (dim + 1)%self.lattice.dimension, 2*ind)
                partition(rng2, (dim + 1)%self.lattice.dimension, 2*ind + 1)
        partition(torch.tensor([[0, self.lattice.size]]*self.lattice.dimension), 0, 1)
        self.wavelet[0] = 1
    
    # construct solution plan for Haar decomposition
    def make_plan(self):
        levmap = self.wavelet.sum(0)
        self.plan = {i:[] for i in range(self.lattice.units)}
        for spot in zip(*torch.nonzero(self.wavelet, as_tuple = True)):
            self.plan[spot[0].item()].append(tuple(x.item() for x in spot[1:]))
        spot2lev = {spot: coordinate_select(levmap, spot).item() for spot in 
                    itertools.product(*[range(d) for d in self.lattice.shape])}
        self.plan = {i: sorted(spots, key=lambda spot: spot2lev[spot]) for i, spots in self.plan.items()}
        
    def _call(self, x):
        wave = self.wavelet.view(torch.Size([1]*(x.dim()-1))+self.wavelet.size())
        x = x.view(x.size() + torch.Size([1]*self.lattice.dimension))
        return self.lattice.group.prod(x * wave, -(self.lattice.dimension+1))

    def _inverse(self, y):
        y = y.clone() # to avoid modifying the original input
        x = torch.zeros(y.size()[:-self.lattice.dimension]+(self.lattice.units,), dtype=torch.long)
        dims = tuple(range(-self.lattice.dimension,0))
        for i, spots in self.plan.items():
            sol = coordinate_select(y, spots[0], dims)
            x[...,i] = sol.squeeze()
            invsol = self.lattice.group.inv(sol)
            for spot in spots[1:]:
                y_spot = coordinate_select(y, spot, dims)
                y_spot.copy_(self.lattice.group.mul(invsol, y_spot))
        return x
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)