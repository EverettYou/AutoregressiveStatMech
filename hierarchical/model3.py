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
        
    def causal_graph(self, speed_of_light: float = 1.):
        """ Construct causal graph 
            Args: speed_of_light - speed of causal cone expansion across one layer
            Returns causal edges, group by their types of causal relations
            in the form of a dictionary
        """
        def discover_causal_connection(z: int):
            # Args: z - level of the source
            source_pos = self.node_centers[2**(z-1):2**z]
            target_pos = self.node_centers[2**z:2**(z+1)]
            displacement = source_pos.unsqueeze(0) - target_pos.unsqueeze(1) # displacement between source and target 
            displacement = (displacement + self.size/2)%self.size - self.size/2 # assuming periodic boundary
            distance = torch.norm(displacement, dim=-1) # distance
            time_scale = 2**((self.tree_depth-1-z)/self.dimension)
            mask = distance < speed_of_light * time_scale
            target_ids, source_ids = torch.nonzero(mask, as_tuple=True)
            return (2**(z-1) + source_ids, 2**z + target_ids)
        def to_adj(edge_index):
            # edge index -> adjecency matrix
            ones = torch.ones(edge_index.size(-1), dtype=torch.long)
            return torch.sparse.LongTensor(edge_index.flip(0), ones, torch.Size([self.sites]*2)).to_dense()
        def to_edge_index(adj):
            # adjacency matrix -> edge index
            target, source = torch.nonzero(adj, as_tuple=True)
            return torch.stack([source, target])
        def re_adj(adj):
            # rectify adjacency matrix:
            # (a) trucate to lower triangle to preserve the causal ordering
            # (b) clamping all elements to 0, 1
            return torch.tril(adj, -1).clamp(0, 1)
        # get direct causal connections
        level_graded_result = [discover_causal_connection(z) for z in range(1, self.tree_depth-1)]
        edge_index = torch.stack([torch.cat(tens, 0) for tens in zip(*level_graded_result)])
        # develop derived causal connections
        adj0 = to_adj(torch.stack([torch.arange(1, self.sites)]*2)) # self (excluding node 0)
        adj1 = to_adj(edge_index) # child
        adj2 = adj1 @ adj1 # grandchild
        #adj3 = adj2 @ adj1 # grandgrandchild
        adj11 = re_adj(adj1 @ adj1.t()) # sibling
        adj22 = re_adj(adj2 @ adj2.t() + adj11) - adj11 # cousin
        adj21 = re_adj(adj2 @ adj1.t() + adj1) - adj1 # niephew
        # collect causal relations by types
        adjs = {'self': adj0, # adjs must at least have 'self' key
                'child': adj1, 
                'sibling': adj11, 
                'niephew': adj21, 
                'cousin': adj22, 
                'grandchild': adj2}
        # convert to edge_index for return
        return {typ: to_edge_index(adjs[typ]) for typ in adjs}

    def node_position_encoding(self):
        """ Construct position encoding of all nodes """
        phase = 2*math.pi*self.node_centers/self.size
        levels = self.node_levels.to(dtype=torch.float).unsqueeze(-1)
        encoding = torch.cat([phase.sin(), phase.cos(), levels], -1)
        return encoding

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
        causal_graph: a dictionary of edges grouped by types
        in_features: number of input features (per node)
        out_features: number of output features (per node)
        bias: whether to learn an edge-depenent bias
        self_loop: whether to include self loops in message passing
    """
    def __init__(self, causal_graph: dict, in_features: int, out_features: int,
                 bias: bool = True, self_loop: bool = True):
        super(GraphConv, self).__init__()
        self.causal_graph = causal_graph.copy() # without copying, update_causal_graph will interfere with each other
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linears = nn.ModuleDict()
        for typ in self.causal_graph:
            self.linears[typ] = nn.Linear(self.in_features, self.out_features, self.bias)
        self.self_loop = self_loop
    
    def extra_repr(self):
        return '(self_loop): {}'.format(self.self_loop)

    def forward(self, input, j = None):
        # input: shape [..., N, in_features]
        # forward from a source node, indexed by j
        # if j is None, forward all nodes
        output = None
        for typ in self.causal_graph:
            if typ is 'self' and not self.self_loop:
                continue # skip 'self' if no self loop
            edge_index = self.causal_graph[typ]
            if j is not None:
                mask = (edge_index[0] == j) # create a mask for edges
                edge_index = edge_index[:, mask]
            typ_output = self.propagate(edge_index, input=input, typ=typ)
            if output is None:
                output = typ_output
            else:
                output += typ_output
        return output

    def propagate(self, edge_index, input, typ):
        # input: shape [..., N, in_features]
        [source, target] = edge_index
        message = self.linears[typ](input[..., source, :])
        output = torch_scatter.scatter_add(message, target,
                    dim = -2, dim_size = input.size(-2))
        return output
    
    def update_causal_graph(self, causal_graph: dict):
        # update causal graph, adding new linear maps if needed
        for typ in causal_graph:
            if typ not in self.causal_graph: #typ is new
                # create a new linear map for it
                self.linears[typ] = nn.Linear(self.in_features, self.out_features, self.bias)
        self.causal_graph.update(causal_graph)
        return self

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
        causal_graph = self.lattice.causal_graph()
        self.layers = nn.ModuleList()
        for l in range(1, len(self.features)):
            if l == 1: # the first layer should not have self loops
                self.layers.append(GraphConv(causal_graph, 
                    self.features[0], self.features[1], bias, self_loop = False))
            else: # remaining layers are normal
                self.layers.append(nn.LayerNorm([self.features[l - 1]]))
                self.layers.append(getattr(nn, nonlinearity)()) # activatioin layer
                self.layers.append(GraphConv(causal_graph,
                    self.features[l - 1], self.features[l], bias))
    
    def forward(self, input):
        output = input
        for layer in self.layers: # apply layers
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

    def update_causal_graph(self, causal_graph: dict):
        # update causal graph for all GraphConv layers
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                layer.update_causal_graph(causal_graph)
        return self

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





















