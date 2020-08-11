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



from phys import Group, LatticeSystem
class HaarTransform(dist.Transform):
    """ Haar wavelet transformation (bijective)
        transformation takes real space configurations x to wavelet space encoding y
    
        Args:
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, group: Group, lattice: LatticeSystem):
        super(HaarTransform, self).__init__()
        self.group = group
        self.lattice = lattice
        self.bijective = True
        self.make_wavelet()
        
    # construct Haar wavelet basis
    def make_wavelet(self):
        self.wavelet = torch.zeros(torch.Size([self.lattice.sites, self.lattice.sites]), dtype=torch.int)
        self.wavelet[0] = 1
        for z in range(1,self.lattice.tree_depth):
            block_size = 2**(z-1)
            for q in range(block_size):
                node_range = 2**(self.lattice.tree_depth-1-z) * torch.tensor([2*q+1,2*q+2])
                nodes = torch.arange(*node_range)
                sites = self.lattice.node_index[nodes]
                self.wavelet[block_size + q, sites] = 1 
                
    def _call(self, x):
        y = self.group.prod(x.unsqueeze(-1) * self.wavelet, -2)
        return y.view(x.size()[:-1]+torch.Size(self.lattice.shape))
    
    def _inverse(self, y):
        x = y.flatten(-self.lattice.dimension)[...,self.lattice.node_index]
        def renormalize(x):
            if x.size(-1) > 1:
                x0 = x[...,0::2]
                x1 = x[...,1::2]
                return torch.cat((renormalize(x0), self.group.mul(self.group.inv(x0), x1)), -1)
            else:
                return x
        return renormalize(x)
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)