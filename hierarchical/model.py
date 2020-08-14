import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import torch_scatter

""" -------- Infrastructures -------- """

class Lattice(object):
    """ Hosts lattice information and construct causal graph
        
        Args:
        size: number of size along one dimension (assuming square/cubical lattice)
        dimension: dimension of the lattice
    """
    def __init__(self, size:int, dimension:int):
        self.size = size
        self.dimension = dimension
        self.shape = [size]*dimension
        self.sites = size**dimension
        self.tree_depth = self.sites.bit_length()
        self.node_init()
        
    def __repr__(self):
        return 'Lattice({} grid with tree depth {})'.format(
                    'x'.join(str(L) for L in self.shape),
                     self.tree_depth)
    
    def node_init(self):
        """ Node initialization, calculate basic node information
            for other methods in this class to work.
            Called by class initialization. """
        #self.node_levels = torch.zeros(self.sites, dtype=torch.int)
        #self.node_centers = torch.zeros(self.sites, self.dimension, dtype=torch.float)
        self.node_index = torch.zeros(self.sites, dtype=torch.long)
        def partition(rng: torch.Tensor, dim: int, ind: int, lev: int):
            if rng[dim].sum()%2 == 0:
                #self.node_levels[ind] = lev
                #self.node_centers[ind] = rng.to(dtype=torch.float).mean(-1)
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
    
    def causal_graph(self, k = 3):
        """ Construct causal graph 
            Args: k - number of generations to consider
        """
        self.family = {} # a dict hosting all relatives
        def child(i, k): # kth-generation child of node i
            return set(2**k * i + q for q in range(2**k))
        def relative(k0, k1): # (k0, k1)-relatives
            # two nodes i0 and i1 are (k0, k1)-relative,
            # if their closest common ancestor is k0 and k1 generations from them respectively
            if (k0, k1) not in self.family: # if relation not found
                rels = set() # start collecting relative relations
                for i in range(1, self.sites//2**max(k0, k1)): # for every possible common ancestor
                    ch0 = child(i, k0) # set of k0-child
                    ch1 = child(i, k1) # set of k1-child
                    rels |= set((i0, i1) for i0 in ch0 for i1 in ch1 if i0 <= i1)
                for k in range(min(k0, k1)): # exlusing closer relatives
                    rels -= relative(k0 - k - 1, k1 - k - 1)
                self.family[(k0, k1)] = rels # record the relations
            return self.family[(k0, k1)]
        # collect all relatives within k generations
        typ = 0
        gen = {}
        for k1 in range(0, k):
            for k0  in range(0, k1 + 1):
                gen[typ] = relative(k0, k1)
                typ += 1
        index_list = [torch.tensor(sorted(list(gen[typ]))).t() for typ in gen]
        type_list = [torch.Tensor().new_full((1, len(gen[typ])), typ, dtype=torch.long) for typ in gen]
        return torch.cat([torch.cat(index_list, -1), torch.cat(type_list, -1)], 0)

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
        self.val_table = None
    
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
            val_table = self.default_val_table()
        elif len(val_table) != self.order:
            raise ValueError('Group function value table must be of the same size as the group order, expect {} got {}.'.format(self.order, len(val_table)))
        return torch.gather(val_table.expand(input.size()[:-1]+(-1,)), -1, input)

    def default_val_table(self):
        if self.val_table is None:
            self.val_table = torch.zeros(self.order)
            self.val_table[0] = 1.
        return self.val_table

class SymmetricGroup(Group):
    """ Represent a permutation group """
    def __init__(self, n: int):
        self.elements = list(itertools.permutations(range(n), n))
        index = {g:i for i, g in enumerate(self.elements)}
        mul_table = torch.empty([len(self.elements)]*2, dtype=torch.long)
        for g1 in self.elements:
            for g2 in self.elements:
                g = tuple(g1[a] for a in g2)
                mul_table[index[g1], index[g2]] = index[g]
        super(SymmetricGroup, self).__init__(mul_table)

    def default_val_table(self):
        if self.val_table is None:
            def cycle_number(g):
                if len(g) == 0:
                    return 0
                elif g[0] == 0:
                    return cycle_number(tuple(a - 1 for a in g[1:])) + 1
                else:
                    return cycle_number(tuple(g[0] - 1 if a == 0 else a - 1 for a in g[1:]))
            self.val_table = torch.tensor([cycle_number(g) for g in self.elements], dtype=torch.float)
        return self.val_table


""" -------- Energy Model -------- """

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
        
    def on(self, group: Group = None, lattice: Lattice = None):
        self.group = group
        self.lattice = lattice
        return self
        
    def forward(self):
        if self.group is None:
            raise RuntimeError('A group structure has not been linked before forward evaluation of the energy term. Call self.on(group = group) to link a Group.')
        if self.lattice is None:
            raise RuntimeError('A lattice system has not been linked before forward evaluation of the energy term. Call self.on(lattice = lattice) to link a Lattice.')

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
    
    def on(self, group: Group = None, lattice: Lattice = None):
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
            return 'G -> {}'.format((self.val_table * self.strength).tolist())
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
            return 'G -> {} across {}'.format(
                (self.val_table * self.strength).tolist(),
                self.shifts if not self.shifts is None else '(0,...)')
        elif not self.shifts is None:
            return '{} across {}'. format(
                self.strength,
                self.shifts)
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
        energy: lattice Hamiltonian in terms of energy terms
        group: a specifying the group on each site
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, energy: EnergyTerms, group: Group, lattice: Lattice):
        super(EnergyModel, self).__init__()
        self.group = group
        self.lattice = lattice
        self.update(energy)
    
    def extra_repr(self):
        return '(group): {}\n(lattice): {}'.format(self.group, self.lattice) + super(EnergyModel, self).extra_repr()
        
    def forward(self, input):
        return self.energy(input)

    def update(self, energy):
        self.energy = energy.on(self.group, self.lattice)

""" -------- Transformations -------- """

class HaarTransform(dist.Transform):
    """ Haar wavelet transformation (bijective)
        transformation takes real space configurations x to wavelet space encoding y
    
        Args:
        group: a group structure for each unit
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, group: Group, lattice: Lattice):
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
                
    def _call(self, z):
        x = self.group.prod(z.unsqueeze(-1) * self.wavelet, -2)
        return x.view(z.size()[:-1]+torch.Size(self.lattice.shape))
    
    def _inverse(self, x):
        y = x.flatten(-self.lattice.dimension)[...,self.lattice.node_index]
        def renormalize(y):
            if y.size(-1) > 1:
                y0 = y[...,0::2]
                y1 = y[...,1::2]
                return torch.cat((renormalize(y0), self.group.mul(self.group.inv(y0), y1)), -1)
            else:
                return y
        z = renormalize(y)
        return z
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)

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

""" -------- Base Distribution -------- """

class GraphConv(nn.Module):
    """ Graph Convolution layer 
        
        Args:
        graph: tensor of shape [3, num_edges] 
               specifying (source, target, type) along each column
        in_features: number of input features (per node)
        out_features: number of output features (per node)
        bias: whether to learn an edge-depenent bias
        self_loop: whether to include self loops in message passing
    """
    def __init__(self, graph: torch.Tensor, in_features: int, out_features: int,
                 bias: bool = True, self_loop: bool = True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = bias
        else:
            self.register_parameter('bias', None)
        self.edge_types = None
        self.update_graph(graph)
        self.self_loop = self_loop

    def update_graph(self, graph):
        # update the graph, adding new linear maps if needed
        self.graph = graph
        edge_types = graph[-1].max() + 1
        if edge_types != self.edge_types:
            self.weight = nn.Parameter(torch.Tensor(edge_types, self.out_features, self.in_features))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.Tensor(edge_types, self.out_features))
            self.reset_parameters()
        self.edge_types = edge_types
        return self

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def extra_repr(self):
        return 'edge_types={}, in_features={}, out_features={}, bias={}, self_loop={}'.format(
            self.edge_types, self.in_features, self.out_features, self.bias is not None, self.self_loop)

    def forward(self, input, j = None):
        # forward from a source node, indexed by j
        # if j is None, forward all nodes
        if j is None: # forward all nodes together
            if self.self_loop: # if self loop allowed
                typ0 = 0 # typ starts from 0
            else: # if self loop forbidden
                typ0 = 1 # typ starts from 1
            output = None
            for typ in range(typ0, self.edge_types):
                mask = (self.graph[2] == typ)
                if output is None:
                    output = self.homo_propagate(self.graph[:2, mask], input, typ)
                else:
                    output += self.homo_propagate(self.graph[:2, mask], input, typ)
        else: # forward from specific node
            graph = self.graph
            mask = (graph[0] == j) # mask out edges from other nodes
            graph = graph[:, mask]
            if not self.self_loop: # no self loop
                mask = (graph[2] != 0) # mask out self loops
                graph = graph[:, mask]
            output = self.hetero_propagate(graph, input)
        return output

    def homo_propagate(self, graph, input, typ):
        [source, target] = graph
        signal = input[..., source, :] # shape [..., E, in_features]
        if self.bias is None:
            message = F.linear(signal, self.weight[typ]) # shape: [..., E, out_features]
        else:
            message = F.linear(signal, self.weight[typ], self.bias[typ]) # shape: [..., E, out_features]
        output = torch_scatter.scatter_add(message, target,
                    dim = -2, dim_size = input.size(-2))
        return output # shape: [..., N, out_features]

    def hetero_propagate(self, graph, input):
        # input: shape [..., N, in_features]
        [source, target, edge_type] = graph
        signal = input[..., source, :] # shape [..., E, in_features]
        weight = self.weight[edge_type] # shape [E, out_features, in_features]
        message = torch.sum(weight * signal.unsqueeze(-2), -1) # shape [..., E, out_features]
        if self.bias is not None:
            bias = self.bias[edge_type] # shape [E, out_features]
            message += bias
        output = torch_scatter.scatter_add(message, target,
                    dim = -2, dim_size = input.size(-2))
        return output # shape: [..., N, out_features]

class AutoregressiveModel(nn.Module, dist.Distribution):
    """ Represent a generative model that can generate samples and evaluate log probabilities.
        
        Args:
        lattice: lattice system
        features: a list of feature dimensions for all layers
        nonlinearity: activation function to use 
        bias: whether to learn the bias
    """
    
    def __init__(self, lattice: Lattice, features, nonlinearity: str = 'Tanh', bias: bool = True):
        super(AutoregressiveModel, self).__init__()
        self.lattice = lattice
        self.nodes = lattice.sites
        self.features = features
        dist.Distribution.__init__(self, event_shape=torch.Size([self.nodes, self.features[0]]))
        self.has_rsample = True
        self.graph = self.lattice.causal_graph()
        self.layers = nn.ModuleList()
        for l in range(1, len(self.features)):
            if l == 1: # the first layer should not have self loops
                self.layers.append(GraphConv(self.graph, self.features[0], self.features[1], bias, self_loop = False))
            else: # remaining layers are normal
                self.layers.append(nn.LayerNorm([self.features[l - 1]]))
                self.layers.append(getattr(nn, nonlinearity)()) # activatioin layer
                self.layers.append(GraphConv(self.graph, self.features[l - 1], self.features[l], bias))

    def update_graph(self, graph):
        # update graph for all GraphConv layers
        self.graph = graph
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                layer.update_graph(graph)
        return self

    def forward(self, input):
        output = input
        for layer in self.layers: # apply layers
            output = layer(output)
        return output # logits
    
    def log_prob(self, sample):
        logits = self(sample) # forward pass to get logits
        return torch.sum(sample * F.log_softmax(logits, dim=-1), (-2,-1))

    def sampler(self, logits, dim=-1): # simplified from F.gumbel_softmax
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels += logits.detach()
        index = gumbels.max(dim, keepdim=True)[1]
        return torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)

    def _sample(self, sample_size: int, sampler = None):
        if sampler is None: # if no sampler specified, use default
            sampler = self.sampler
        # create a list of tensors to cache layer-wise outputs
        cache = [torch.zeros(sample_size, self.nodes, self.features[0])]
        for layer in self.layers:
            if isinstance(layer, GraphConv): # for graph convolution layers
                features = layer.out_features # features get updated
            cache.append(torch.zeros(sample_size, self.nodes, features))
        # cache established. start by sampling node 0.
        # assuming global symmetry, node 0 is always sampled uniformly
        cache[0][..., 0, :] = sampler(cache[0][..., 0, :])
        # start autoregressive sampling
        for j in range(1, self.nodes): # iterate through nodes 1:all
            for l, layer in enumerate(self.layers):
                if isinstance(layer, GraphConv): # for graph convolution layers
                    if l==0: # first layer should forward from previous node
                        cache[l + 1] += layer(cache[l], j - 1)
                    else: # remaining layers forward from this node
                        cache[l + 1] += layer(cache[l], j)
                else: # for other layers, only update node j (other nodes not ready yet)
                    src = layer(cache[l][..., [j], :])
                    index = torch.tensor(j).view([1]*src.dim()).expand(src.size())
                    cache[l + 1] = cache[l + 1].scatter(-2, index, src)
            # the last cache hosts the logit, sample from it 
            cache[0][..., j, :] = sampler(cache[-1][..., j, :])
        return cache # cache[0] hosts the sample
    
    def sample(self, sample_size=1):
        with torch.no_grad():
            cache = self._sample(sample_size)
        return cache[0]
    
    def rsample(self, sample_size=1, tau=None, hard=False):
        # reparametrized Gumbel sampling
        if tau is None: # if temperature not given
            tau = 1/(self.features[-1]-1) # set by the out feature dimension
        cache = self._sample(sample_size, lambda x: F.gumbel_softmax(x, tau, hard))
        return cache[0]

    def sample_with_log_prob(self, sample_size=1):
        cache = self._sample(sample_size)
        sample = cache[0]
        logits = cache[-1]
        log_prob = torch.sum(sample * F.log_softmax(logits, dim=-1), (-2,-1))
        return sample, log_prob


""" -------- Model Interface -------- """

class HolographicPixelGCN(nn.Module, dist.TransformedDistribution):
    """ Combination of hierarchical autoregressive and flow-based model for lattice models.
    
        Args:
        energy: a energy model to learn
        hidden_features: a list of feature dimensions of hidden layers
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    def __init__(self, energy: EnergyModel, hidden_features, nonlinearity: str = 'Tanh', bias: bool = True):
        super(HolographicPixelGCN, self).__init__()
        self.energy = energy
        self.group = energy.group
        self.lattice = energy.lattice
        self.haar = HaarTransform(self.group, self.lattice)
        self.onecat = OneHotCategoricalTransform(self.group.order)
        features = [self.group.order] + hidden_features + [self.group.order]
        auto = AutoregressiveModel(self.lattice, features, nonlinearity, bias)
        dist.TransformedDistribution.__init__(self, auto, [self.onecat, self.haar])
        self.transform = dist.ComposeTransform(self.transforms)





















