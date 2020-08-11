import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math
class LatticeSystem(object):
    """ host lattice information and construct graph in hyperbolic space
        
        Args:
        size: number of size along one dimension (assuming square/cubical lattice)
        dimension: dimension of the lattice
        causal_radius: radius of the causal cone across one level 
        scale_resolved: whether to distinguish edges from different levels
    """
    def __init__(self, size:int, dimension:int, causal_radius: float = 1., scale_resolved: bool = True):
        self.size = size
        self.dimension = dimension
        self.shape = [size]*dimension
        self.sites = size**dimension
        self.tree_depth = self.sites.bit_length()
        self.node_init()
        self.reset_causal_graph(causal_radius, scale_resolved)
        
    def __repr__(self):
        return 'LatticeSystem({} grid with tree depth {}\n\t(node_index): {}\n\t(edge_index): {}\n\t(edge_type): {})'.format('x'.join(str(L) for L in self.shape), self.tree_depth, self.node_index, self.edge_index, self.edge_type)
    
    def node_init(self):
        self.node_levels = torch.zeros(self.sites, dtype=torch.int)
        self.node_centers = torch.zeros(self.sites, self.dimension, dtype=torch.float)
        self.node_index = torch.zeros(self.sites, dtype=torch.long)
        def partition(rng: torch.Tensor, dim: int, ind: int, lev: int):
            if rng[dim].sum()%2 == 0:
                self.node_levels[ind] = lev
                self.node_centers[ind] = rng.to(dtype=torch.float).mean(-1)
                mid = rng[dim].sum()//2
                rng1 = rng.clone()
                rng1[dim, 1] = mid
                rng2 = rng.clone()
                rng2[dim, 0] = mid
                partition(rng1, (dim + 1)%self.dimension, 2*ind, lev+1)
                partition(rng2, (dim + 1)%self.dimension, 2*ind + 1, lev+1)
            else:
                self.node_index[ind-self.sites] = rng[:,0].dot(self.size**torch.arange(0,self.dimension).flip(0))
        partition(torch.tensor([[0, self.size]]*self.dimension), 0, 1, 1)
        
    def reset_causal_graph(self, causal_radius: float, scale_resolved: bool = True):
        def discover_causal_connection(z: int):
            # Args: z - level of the source
            source_pos = self.node_centers[2**(z-1):2**z]
            target_pos = self.node_centers[2**z:2**(z+1)]
            diff = source_pos.unsqueeze(0) - target_pos.unsqueeze(1)
            diff = (diff + self.size/2)%self.size - self.size/2
            dist = torch.norm(diff, dim=-1)
            smooth_scale = 2**((self.tree_depth-1-z)/self.dimension)
            mask = dist < causal_radius * smooth_scale
            target_ids, source_ids = torch.nonzero(mask, as_tuple=True)
            step_scale = 2**math.floor((self.tree_depth-1-z)/self.dimension)
            edge_signatures = torch.round(2*diff/step_scale)[target_ids, source_ids].to(dtype=torch.int)
            level_signatures = torch.tensor([[z]]*len(source_ids))
            if scale_resolved:
                signatures = torch.cat((level_signatures, edge_signatures), -1)
            else:
                signatures = edge_signatures
            return (2**(z-1) + source_ids, 2**z + target_ids, signatures)
        level_graded_result = [discover_causal_connection(z) for z in range(1, self.tree_depth-1)]
        source_ids, target_ids, signatures = [torch.cat(tens, 0) for tens in zip(*level_graded_result)]
        signatures = [tuple(signature) for signature in signatures.tolist()]
        distinct_signatures = set(signatures)
        self.edge_type_map = {signature: i + 1 for i, signature in enumerate(distinct_signatures)}
        self.edge_type = torch.tensor([self.edge_type_map[signature] for signature in signatures])
        self.edge_index = torch.stack((source_ids, target_ids), 0)
        return self.edge_index, self.edge_type
    
    
class Group(object):
    """Represent a group, providing multiplication and inverse operation.
    
    Args:
    mul_table: multiplication table as a tensor, e.g. Z2 group: tensor([[0,1],[1,0]])
    """
    def __init__(self, mul_table: torch.Tensor):
        super(Group, self).__init__()
        self.mul_table = mul_table
        self.order = mul_table.size(0) # number of group elements
        gs, ginvs = torch.nonzero(self.mul_table == 0, as_tuple=True)
        self.inv_table = torch.gather(ginvs, 0, gs)
    
    def __iter__(self):
        return iter(range(self.order))
    
    def __repr__(self):
        return 'Group({} elements)'.format(self.order)
    
    def inv(self, input: torch.Tensor):
        return torch.gather(self.inv_table.expand(input.size()[:-1]+(-1,)), -1, input)
    
    def mul(self, input1: torch.Tensor, input2: torch.Tensor):
        output = input1 * self.order + input2
        return torch.gather(self.mul_table.flatten().expand(output.size()[:-1]+(-1,)), -1, output)
    
    def prod(self, input, dim: int, keepdim: bool = False):
        input_size = input.size()
        flat_mul_table = self.mul_table.flatten().expand(input_size[:dim]+input_size[dim+1:-1]+(-1,))
        output = input.select(dim, 0)
        for i in range(1, input.size(dim)):
            output = output * self.order + input.select(dim, i)
            output = torch.gather(flat_mul_table, -1, output)
        if keepdim:
            output = output.unsqueeze(dim)
        return output
    
    def val(self, input, val_table = None):
        if val_table is None:
            val_table = torch.zeros(self.order)
            val_table[0] = 1.
        elif len(val_table) != self.order:
            raise ValueError('Group function value table must be of the same size as the group order, expect {} got {}.'.format(self.order, len(val_table)))
        return torch.gather(val_table.expand(input.size()[:-1]+(-1,)), -1, input)



class EnergyTerm(nn.Module):
    """ represent an energy term"""
    strength = 1.
    group = None
    lattice = None
    def __init__(self):
        super(EnergyTerm, self).__init__()
        
    def __mul__(self, other):
        self.strength *= other
        return self
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def __add__(self, other):
        if isinstance(other, EnergyTerm):
            return EnergyTerms([self, other])
        elif isinstance(other, EnergyTerms):
            return other.append(self)
        
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (- other)
    
    def __rsub__(self, other):
        return (- self) + other
    
    def extra_repr(self):
        return '{}'.format(self.strength)
        
    def on(self, group: Group = None, lattice: LatticeSystem = None):
        self.group = group
        self.lattice = lattice
        return self
        
    def forward(self):
        if self.group is None:
            raise RuntimeError('A group structure has not been linked before forward evaluation of the energy term. Call self.on(group = group) to link a Group.')
        if self.lattice is None:
            raise RuntimeError('A lattice system has not been linked before forward evaluation of the energy term. Call self.on(lattice = lattice) to link a LatticeSystem.')

class EnergyTerms(nn.ModuleList):
    """ represent a sum of energy terms"""
    def __init__(self, *arg):
        super(EnergyTerms, self).__init__(*arg)
    
    def __mul__(self, other):
        for term in self:
            term = term * other
        return self
        
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def on(self, group: Group = None, lattice: LatticeSystem = None):
        for term in self:
            term.on(group, lattice)
        return self
    
    def forward(self, input):
        return sum(term(input) for term in self)

class OnSite(EnergyTerm):
    """ on-site energy term """
    def __init__(self, val_table = None):
        super(OnSite, self).__init__()
        self.val_table = val_table
    
    def extra_repr(self):
        if not self.val_table is None:
            return '{}'.format(self.val_table * self.strength)
        else:
            return super(OnSite, self).extra_repr()
    
    def forward(self, input):
        super(OnSite, self).forward()
        dims = tuple(range(-self.lattice.dimension,0))
        energy = self.group.val(input, self.val_table) * self.strength
        return energy.sum(dims)
    
class TwoBody(EnergyTerm):
    """ two-body interaction term """
    def __init__(self, val_table = None, shifts = None):
        super(TwoBody, self).__init__()
        self.val_table = val_table
        self.shifts = shifts
        
    def extra_repr(self):
        if not self.val_table is None:
            return '{} across {}'.format(self.val_table * self.strength, self.shifts if not self.shifts is None else '(0,...)')
        else:
            return super(TwoBody, self).extra_repr()
        
    def forward(self, input):
        super(TwoBody, self).forward()
        dims = tuple(range(-self.lattice.dimension,0))
        if self.shifts is None:
            self.shifts = (0,)*self.lattice.dimension
        rolled = self.group.inv(input.roll(self.shifts, dims))
        coupled = self.group.mul(rolled, input)
        energy = self.group.val(coupled, self.val_table) * self.strength
        return energy.sum(dims)

    
class EnergyModel(nn.Module):
    """ Energy mdoel that describes the physical system. Provides function to evaluate energy.
    
        Args:
        group: a specifying the group on each site
        lattice: a lattice system containing information of the group and lattice shape
        energy: lattice Hamiltonian in terms of energy terms
    """
    def __init__(self, group: Group, lattice: LatticeSystem, energy: EnergyTerms):
        super(EnergyModel, self).__init__()
        self.group = group
        self.lattice = lattice
        self.energy = energy.on(self.group, self.lattice)
    
    def extra_repr(self):
        return '(lattice): {}'.format(self.lattice) + super(EnergyModel, self).extra_repr()
        
    def forward(self, input):
        return self.energy(input)
    
    def update(self, energy: EnergyTerms):
        self.energy = energy.on(self.group, self.lattice)
        return self