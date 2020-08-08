import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

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


class LatticeSystem(object):
    """ a container to host lattice information
        
        Args:
        group: a group that defines multiplication among elements
        size: length of the lattcie along one dimension (should be a power of 2)
        dimension: dimension of the lattice
    """
    def __init__(self, group: Group, size: int, dimension: int):
        super(LatticeSystem, self).__init__()
        self.group = group
        self.size = size
        self.dimension = dimension
        self.shape = [size]*dimension
        self.units = size**dimension
        
    def __repr__(self):
        return 'LatticeSystem({} on {} grid)'.format(self.group, 'x'.join(str(L) for L in self.shape))
    

class EnergyTerm(nn.Module):
    """ represent an energy term"""
    strength = 1.
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
        
    def on(self, lattice: LatticeSystem):
        self.lattice = lattice
        return self
        
    def forward(self):
        if self.lattice is None:
            raise RuntimeError('A lattice system has not been linked before forward evaluation of the energy term. Call self.on(lattice) to link a LatticeSystem.')

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
    
    def on(self, lattice: LatticeSystem):
        for term in self:
            term.on(lattice)
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
        energy = self.lattice.group.val(input, self.val_table) * self.strength
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
        rolled = self.lattice.group.inv(input.roll(self.shifts, dims))
        coupled = self.lattice.group.mul(rolled, input)
        energy = self.lattice.group.val(coupled, self.val_table) * self.strength
        return energy.sum(dims)

    
class EnergyModel(nn.Module):
    """ Energy mdoel that describes the physical system. Provides function to evaluate energy.
    
        Args:
        lattice: a lattice system containing information of the group and lattice shape
        energy: lattice Hamiltonian in terms of energy terms
    """
    def __init__(self, lattice: LatticeSystem, energy: EnergyTerms):
        super(EnergyModel, self).__init__()
        self.lattice = lattice
        self.energy = energy.on(lattice)
    
    def extra_repr(self):
        return '(lattice): {}'.format(self.lattice) + super(EnergyModel, self).extra_repr()
        
    def forward(self, input):
        return self.energy(input)