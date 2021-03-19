import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

""" -------- Graph -------- """

class Graph(object):
    """ Host graph information and enables graph expansion
        
        Args:
        dims: (target_dim, source_dim) number of target/source variables
        indices: index list of adjacency matrix [2, edge_num]
        edge_types: edge type list of adjacency matrix [edge_num]
        source_depths (optional): depth assignments of source variables
    """
    def __init__(self, dims:int or tuple, indices, edge_types, source_depths=None):
        if isinstance(dims, int):
            self.dims = (dims, dims)
        elif isinstance(dims, tuple):
            self.dims = dims
        self.indices = indices
        self.edge_types = edge_types
        if len(self.edge_types) == 0:
            self.max_edge_type = 0
        else:
            self.max_edge_type = edge_types.max().item()
        if source_depths is None:
            self.source_depths = self.get_depth_assignment()
        else:
            self.source_depths = source_depths
        self.edge_depths = self.source_depths[self.indices[1,:]]
        self.max_depth = self.source_depths.max().item()
        
    def __repr__(self):
        return 'Graph({}, {} edges of {} types)'.format('x'.join(str(v) for v in self.dims), self.edge_types.shape[0], self.max_edge_type)
    
    def adjacency_matrix(self):
        return torch.sparse_coo_tensor(self.indices, self.edge_types, self.dims)
    
    def get_depth_assignment(self):
        assert self.dims[0] == self.dims[1], 'get_depths can only be called with square adjacency matrix.'
        dvec = torch.zeros(self.dims[0], dtype=torch.long)
        uvec = torch.ones(self.dims[0], dtype=torch.long)
        adjmat = self.adjacency_matrix()
        while True:
            uvec_new = (adjmat @ uvec > 0).long()
            if uvec_new.sum() == uvec.sum():
                break
            uvec = uvec_new
            dvec += uvec
        if uvec.sum() != 0: # there are nodes trapped in loops
            raise Warning('When assigning depth, discover the following vertices trapped in loops: {}'.format(torch.nonzero(uvec,as_tuple=True)[0].tolist()))
        return dvec
    
    def add_self_loops(self, start = 0):
        # start: the sarting node from which on the self-loop should be added
        assert self.dims[0] == self.dims[1], 'add_self_loops can only be called with square adjacency matrix.'
        loops = torch.arange(start, self.dims[0])
        indices_prepend = torch.stack([loops, loops])
        edge_types_prepend = torch.ones(loops.shape, dtype=torch.long)
        indices = torch.cat([indices_prepend, self.indices], -1)
        edge_types = torch.cat([edge_types_prepend, self.edge_types+1], -1)
        return Graph(self.dims, indices, edge_types, self.source_depths)
    
    def expand(self, target_dim, source_dim):
        # prepare views
        indices = self.indices.view(2,-1,1,1)
        edge_types = self.edge_types.view(-1,1,1)
        target_inds = torch.arange(target_dim).view(-1,1)
        source_inds = torch.arange(source_dim).view(1,-1)
        # calculate indices extension
        target_inds_ext = indices[0,...] * target_dim + target_inds
        source_inds_ext = indices[1,...] * source_dim + source_inds
        # calculate edge type extension
        edge_types_ext = ((edge_types - 1) * target_dim + target_inds) * source_dim + source_inds + 1
        # expand and flatten tensor
        target_inds_ext = target_inds_ext.expand(edge_types_ext.shape).flatten()
        source_inds_ext = source_inds_ext.expand(edge_types_ext.shape).flatten()
        edge_types_ext = edge_types_ext.flatten()
        # expand depths (to the source side)
        source_depths_ext = self.source_depths.repeat_interleave(source_dim)
        dims_ext = (self.dims[0] * target_dim, self.dims[1] * source_dim)
        indices_ext = torch.stack([target_inds_ext, source_inds_ext])
        return Graph(dims_ext, indices_ext, edge_types_ext, source_depths_ext)
    
    def sparse_matrix(self, vector, depth = None):
        if depth is None:
            indices = self.indices
            edge_types = self.edge_types
        else:
            select = self.edge_depths == depth
            indices = self.indices[:, select]
            edge_types = self.edge_types[select]
        return torch.sparse_coo_tensor(indices, vector[edge_types-1], self.dims)

""" -------- Lattice -------- """

class Node(object):
    """ Represent a node object, containing coordinate and relationship information.
    """
    def __init__(self, ind:int):
        self.type = None
        self.ind = ind
        self.center = None
        self.generation = None
        self.parent = None
        self.children = [None, None]
        self.site = None
        
    def __repr__(self):
        return 'Node({})'.format(self.ind)
    
    def ancestors(self):
        # ancestor = self + ancestor of parent
        if self.parent is not None:
            return [self] + self.parent.ancestors()
        else:
            return []
    
    def shadow_sites(self):
        # shadow_sites = sum of shadow_sites of children
        if self.type is 'lat':
            shd = []
            for node in self.children:
                shd += node.shadow_sites()
            return shd
        elif self.type is 'phy':
            return [self.site]
        
    def action_sites(self):
        # action_sites = shadow_sites of last child
        if self.type is 'lat':
            return self.children[-1].shadow_sites()
        elif self.type is 'phy':
            return []

class Lattice(object):
    """ Host lattice information and construct causal graph
        
        Args:
        size: number of size along one dimension (assuming square/cubical lattice)
        dimension: dimension of the lattice
    """
    def __init__(self, size:int, dimension:int):
        assert size > 0, "lattice size must be a positive integer."
        assert dimension > 0, "lattice dimension must be a positive integer."
        self.size = size
        self.dimension = dimension
        self.sites = size**dimension
        self.nodes = []

    def __repr__(self):
        return 'Lattice({} grid)'.format('x'.join(str(self.size) for k in range(self.dimension)))

    def relevant_nodes(self, node, **kwargs):
        raise NotImplementedError

    def relationship(self, node1, node2, **kwargs):
        raise NotImplementedError

    def causal_graph(self, **kwargs):
        relations = set()
        edges = {}
        for target_node in self.nodes[1:]:
            if target_node.type is 'lat':
                for source_node in self.relevant_nodes(target_node, **kwargs):
                    relation = self.relationship(source_node, target_node, **kwargs)
                    relations.add(relation)
                    edges[(target_node.ind, source_node.ind)] = relation
        relations = list(relations)
        type_map = {relation: k+1 for k, relation in enumerate(relations)}
        indices = torch.zeros((2, len(edges)), dtype = torch.long)
        edge_types = torch.zeros(len(edges), dtype = torch.long)
        for k, (edge, relation) in enumerate(edges.items()):
            indices[0, k] = edge[0]
            indices[1, k] = edge[1]
            edge_types[k] = type_map[relation]
        graph = Graph(self.sites, indices, edge_types)
        graph.type_dict = {edge_type: relation for relation, edge_type in type_map.items()}
        return graph
    
    def randperm(self, sample_size):
        grid = torch.tensor(list(itertools.product(range(self.size), repeat=self.dimension)))
        new_grids = []
        for _ in range(sample_size):
            new_grid = grid[:,torch.randperm(self.dimension)]
            new_grid *= (-1)**torch.randint(2, (1,self.dimension))
            new_grid += torch.randint(self.size, (1, self.dimension))
            new_grid = new_grid % self.size
            new_grids.append(new_grid)
        base = self.size**torch.arange(self.dimension).flip(0)
        perms = torch.tensordot(torch.stack(new_grids), base, dims=1)
        return perms

class FlatLattice(Lattice):
    """ Flat lattice (regular grid in flat space) """
    def __init__(self, size:int, dimension:int):
        super(FlatLattice, self).__init__(size, dimension)
        self.nodes = [Node(i) for i in range(self.sites)]
        for i, center in enumerate(itertools.product(range(self.size), repeat=self.dimension)):
            self.nodes[i].type = 'lat'
            self.nodes[i].center = torch.tensor(center).float()
            
    def relevant_nodes(self, node, radius=2., **kwargs):
        relevant_nodes = set()
        for prior_node in self.nodes[:node.ind]:
            displacement = prior_node.center - node.center
            displacement = (displacement + self.size/2)%self.size - self.size/2
            if displacement.norm() < radius:
                relevant_nodes.add(prior_node)
        return relevant_nodes
    
    def relationship(self, node1, node2, **kwargs):
        displacement = node1.center - node2.center
        displacement = (displacement + self.size/2)%self.size - self.size/2
        symmetrized, _ = displacement.abs().sort()
        return tuple(symmetrized.long().tolist())
    
class TreeLattice(Lattice):
    """ Tree lattice (binary H-tree, latent nodes + physical nodes) """
    def __init__(self, size:int, dimension:int):
        assert (size & (size-1) == 0), "lattice size must be a power of 2 for TreeLattice."
        super(TreeLattice, self).__init__(size, dimension)
        self.nodes = [Node(i) for i in range(2*self.sites)]
        self.nodes[0].type = 'lat'
        self.nodes[0].generation = 0
        self.nodes[0].children = [self.nodes[1]]
        self.nodes[1].parent = self.nodes[0]
        def partition(rng: torch.Tensor, dim: int, ind: int, gen: int):
            this_node = self.nodes[ind]
            this_node.center = rng.float().mean(-1)
            this_node.generation = gen
            if rng[dim].sum()%2 == 0:
                this_node.type = 'lat'
                mid = rng[dim].sum()//2
                rng1 = rng.clone()
                rng1[dim, 1] = mid
                rng2 = rng.clone()
                rng2[dim, 0] = mid
                ind1 = (ind-2**gen)+2*2**gen
                ind2 = (ind-2**gen)+3*2**gen
                partition(rng1, (dim + 1)%self.dimension, ind1, gen+1)
                partition(rng2, (dim + 1)%self.dimension, ind2, gen+1)
                this_node.children = [self.nodes[ind1], self.nodes[ind2]]
                self.nodes[ind1].parent = this_node
                self.nodes[ind2].parent = this_node
            else:
                this_node.type = 'phy'
                this_node.site = rng[:,0].dot(self.size**torch.arange(0,self.dimension).flip(0)).item()
        partition(torch.tensor([[0, self.size]]*self.dimension), 0, 1, 0)
            
    def wavelet_maps(self):
        decoder_map = torch.zeros((self.sites,self.sites), dtype=torch.long)
        for node in self.nodes:
            if node.type is 'lat':
                source = node.ind
                for target in node.action_sites():
                    decoder_map[target, source] = 1
        encoder_map = torch.inverse(decoder_map.double()).round().long()
        return encoder_map, decoder_map
                    
    def relevant_nodes(self, node, radius = 1., **kwargs):
        # relevant_nodes = union of ancestors of adjacent nodes within given radius
        scaled_radius = radius * self.size / 2**(node.generation/self.dimension)
        relevant_nodes = set()
        for prior_node in self.nodes[1:node.ind]:
            displacement = prior_node.center - node.center
            displacement = (displacement + self.size/2)%self.size - self.size/2
            if displacement.norm() < scaled_radius:
                relevant_nodes.update(prior_node.ancestors())
        return relevant_nodes
    
    def common_ancestor(self, node1, node2):
        # the closest common ancestor of two nodes
        common_ancestor = None
        while common_ancestor is None:
            if node1.generation == node2.generation:
                if node1 is node2:
                    common_ancestor = node1
                else:
                    node1 = node1.parent
                    node2 = node2.parent
            elif node1.generation < node2.generation:
                node2 = node2.parent
            else: # node1.generation > node2.generation
                node1 = node1.parent
        return common_ancestor
    
    def relationship(self, node1, node2, scale_invariance=True, **kwargs):
        gen1, gen2 = node1.generation, node2.generation
        gen0 = self.common_ancestor(node1, node2).generation
        if scale_invariance:
            return (gen1 - gen0, gen2 - gen0)
        else:
            return (gen0, gen1 - gen0, gen2 - gen0)

""" -------- Group -------- """

class Group(nn.Module):
    """Represent a group, providing multiplication and inverse operation.
    
    Args:
    mul_table: multiplication table as a tensor, e.g. Z2 group: tensor([[0,1],[1,0]])
    """
    def __init__(self, mul_table: torch.Tensor):
        super(Group, self).__init__()
        self.mul_table = mul_table
        self.order = mul_table.shape[0] # number of group elements
        gs, ginvs = torch.nonzero(self.mul_table == 0, as_tuple=True)
        self.inv_table = torch.gather(ginvs, 0, gs)
    
    def __iter__(self):
        return iter(range(self.order))
    
    def __repr__(self):
        return 'Group(order={})'.format(self.order)
    
    def inv(self, input: torch.Tensor):
        return self.inv_table.to(input.device)[input]
    
    def mod(self, input: torch.Tensor):
        output = input
        output[output < 0] = self.inv(output[output < 0].abs())
        return output
    
    def mul(self, input1: torch.Tensor, input2: torch.Tensor):
        return self.mul_table.flatten().to(input1.device)[input1 * self.order + input2]
    
    def prod(self, input, dim: int, keepdim: bool = False):
        output = input.select(dim, 0)
        for i in range(1, input.shape[dim]):
            output = self.mul(output, input.select(dim, i))
        if keepdim:
            output = output.unsqueeze(dim)
        return output
    
    def forward(self, input, val_table):
        assert len(val_table) == self.order, 'Group function value table must be of the same size as the group order, expect {} got {}.'.format(self.order, len(val_table))
        return val_table.to(input.device)[input]

    def default_val_table(self):
        val_table = torch.zeros(self.order)
        val_table[0] = 1.
        return val_table

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
        def cycle_number(g):
            if len(g) == 0:
                return 0
            elif g[0] == 0:
                return cycle_number(tuple(a - 1 for a in g[1:])) + 1
            else:
                return cycle_number(tuple(g[0] - 1 if a == 0 else a - 1 for a in g[1:]))
        val_table = torch.tensor([cycle_number(g) for g in self.elements], dtype=torch.float)
        return val_table

""" -------- Energy Model -------- """

class EnergyTerm(nn.Module):
    """ represent an energy term"""
    def __init__(self, val_table = None):
        super(EnergyTerm, self).__init__()
        self.strength = 1.
        if val_table is None:
            self.val_table = None
        else:
            self.val_table = torch.tensor(val_table)
        
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
        
    def forward(self, lattice, group):
        if self.val_table is None:
            self.val_table = group.default_val_table()

class OnSite(EnergyTerm):
    """ on-site energy term """
    def __init__(self, val_table = None):
        super(OnSite, self).__init__(val_table)
    
    def extra_repr(self):
        if not self.val_table is None:
            return 'G -> {}'.format((self.val_table * self.strength).tolist())
        else:
            return super(OnSite, self).extra_repr()
    
    def forward(self, input, lattice, group):
        super(OnSite, self).forward(lattice, group)
        dims = tuple(range(-lattice.dimension,0))
        energy = group(input, self.val_table) * self.strength
        return energy.sum(dims)
    
class TwoBody(EnergyTerm):
    """ two-body interaction term """
    def __init__(self, shifts = None, val_table = None):
        super(TwoBody, self).__init__(val_table)
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
        
    def forward(self, input, lattice, group):
        super(TwoBody, self).forward(lattice, group)
        dims = tuple(range(-lattice.dimension,0))
        if self.shifts is None:
            self.shifts = [0]*lattice.dimension
        rolled = group.inv(input.roll(self.shifts, dims))
        coupled = group.mul(rolled, input)
        energy = group(coupled, self.val_table) * self.strength
        return energy.sum(dims)
    
class EnergyTerms(nn.ModuleList):
    """ represent a sum of energy terms"""
    def __init__(self, *args):
        super(EnergyTerms, self).__init__(*args)
    
    def __mul__(self, other):
        for term in self:
            term = term * other
        return self
        
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def forward(self, input, lattice, group):
        return sum(term(input, lattice, group) for term in self)

class Model(nn.Module):
    """ Energy mdoel that describes the physical system. Provides function to evaluate energy.
    
        Args:
        energy: lattice Hamiltonian in terms of energy terms
        group: a specifying the group on each site
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, energy: EnergyTerms, lattice: Lattice, group: Group):
        super(Model, self).__init__()
        self.lattice = lattice
        self.group = group
        self.energy = energy
    
    def extra_repr(self):
        return '(lattice): {}'.format(self.lattice)
        
    def forward(self, input):
        return self.energy(input, self.lattice, self.group)

""" -------- Transformations -------- """

class ReshapeTransform(dist.Transform):
    """ Arrange flatten variables on the lattice and vice versa

        Args:
        lattice: a flat lattice system
    """
    def __init__(self, lattice: Lattice):
        super(ReshapeTransform, self).__init__()
        self.lattice = lattice
        self.bijective = True

    def _call(self, x):
        return x.view(x.shape[:-1]+(self.lattice.size,)*self.lattice.dimension)

    def _inverse(self, x):
        return x.flatten(-self.lattice.dimension)

class HaarTransform(dist.Transform):
    """ Haar wavelet transformation (bijective)
        transformation takes real space configurations x to wavelet space encoding y
    
        Args:
        lattice: a tree lattice system
        group: a group structure for each variable
    """
    def __init__(self, lattice: Lattice, group: Group):
        super(HaarTransform, self).__init__()
        self.lattice = lattice
        self.group = group
        self.bijective = True
        self.encoding_mat, self.decoding_mat = self.lattice.wavelet_maps()
                
    def _call(self, z):
        x = self.group.prod(z.unsqueeze(-2) * self.decoding_mat.to(z.device), -1)
        return x.view(z.shape[:-1]+(self.lattice.size,)*self.lattice.dimension)
    
    def _inverse(self, x): 
        x = x.flatten(-self.lattice.dimension)
        return self.group.prod(self.group.mod(x.unsqueeze(-2) * self.encoding_mat.to(x.device)), -1)
    
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
        return F.one_hot(y, self.num_classes).float()
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)

""" -------- Graph Convolution -------- """

class GraphConvLayer(nn.Module):
    """ Graph Convolution layer 
        
        Args:
        graph: graph object
        in_features: number of input features (per node)
        out_features: number of output features (per node)
        bias: (optional) whether to learn an edge-depenent bias
        self_loop: (optional) whether to include self loops in message passing
    """
    def __init__(self, 
                 graph:Graph,
                 in_features:int,
                 out_features:int,
                 bias:bool = True,
                 self_loop:bool or int = True, 
                 **kwargs):
        super(GraphConvLayer, self).__init__()
        self.self_loop = self_loop
        if isinstance(self.self_loop, bool):
            if self.self_loop:
                self.graph = graph.add_self_loops()
            else:
                self.graph = graph
        else:
            self.graph = graph.add_self_loops(start=self.self_loop)
        self.in_features = in_features
        self.out_features = out_features
        self.weight_graph = self.graph.expand(self.out_features, self.in_features)
        self.weight_vector = nn.Parameter(torch.Tensor(self.weight_graph.max_edge_type))
        self.bias = bias
        if self.bias:
            self.bias_graph = self.graph.expand(self.out_features, 1)
            self.bias_vector = nn.Parameter(torch.Tensor(self.bias_graph.max_edge_type))        
        self.reset_parameters()
        (self.target_dim, self.source_dim) = self.weight_graph.dims
    
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_vector, -bound, bound)
        if self.bias:
            nn.init.uniform_(self.bias_vector, -bound, bound)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, self_loop={}\n{}'.format(
            self.in_features, self.out_features, self.bias, self.self_loop, self.graph)
    
    def forward(self, x, depth=None):
        weight_matrix = self.weight_graph.sparse_matrix(self.weight_vector, depth)
        y = torch.sparse.mm(weight_matrix, x)
        if self.bias:
            bias_matrix = self.bias_graph.sparse_matrix(self.bias_vector, depth)
            unit = torch.ones((bias_matrix.shape[1], 1), dtype=bias_matrix.dtype, device=bias_matrix.device)
            y = y + torch.sparse.mm(bias_matrix, unit)
        return y
    
class GraphConvNet(nn.Module):
    """ Graph Convolution network 
        
        Args:
        graph: graph object
        features: a list of numbers of features (per node) across layers
        nonlinearity: (optional) nonlinear activation to use
    """
    def __init__(self, 
                 graph:Graph, 
                 features, 
                 nonlinearity:str = 'ReLU', 
                 **kwargs):
        super(GraphConvNet, self).__init__()
        self.graph = graph
        self.features = features
        self.layers = nn.ModuleList()
        for l in range(1, len(self.features)):
            if l == 1: # the first layer should not have self loops
                self.layers.append(GraphConvLayer(self.graph, self.features[0], self.features[1], self_loop=False, **kwargs))
            else: # remaining layers are normal
                self.layers.append(getattr(nn, nonlinearity)()) # activatioin layer
                self.layers.append(GraphConvLayer(self.graph, self.features[l - 1], self.features[l], self_loop=1, **kwargs))
                
    def forward(self, input, depth=None, cache=None):
        # input: [..., nodes, features]
        in_shape = input.shape
        batch_dim = torch.tensor(in_shape[:-2]).prod()
        input_dim = torch.tensor(in_shape[-2:]).prod()
        x = input.view((batch_dim, input_dim)).T
        if depth is None:
            for layer in self.layers:
                x = layer(x)
        else: # depth-specific forward
            if cache is None: # if cache not exist, prepare cache
                cache = [x]
                for layer in self.layers:
                    if isinstance(layer, GraphConvLayer):
                        target_dim = layer.target_dim
                    cache.append(torch.zeros((target_dim, batch_dim), device=x.device))
            else: # if cache exist, load x to cache[0]
                cache[0] = x
            # cache is ready, start forwarding
            for l, layer in enumerate(self.layers):
                if isinstance(layer, GraphConvLayer):
                    if l == 0: # first layer should forward from the previous depth
                        cache[l+1] = cache[l+1] + layer(cache[l], depth - 1)
                    else: # remaining layer forward from the current depth
                        cache[l+1] = cache[l+1] + layer(cache[l], depth)
                else:
                    cache[l+1] = layer(cache[l])
            x = cache[-1] # last cache hosts output
        out_shape = in_shape[:-1]+(self.features[-1],)
        output = x.T.view(out_shape)
        if cache is None:
            return output
        else:
            return output, cache

""" -------- Base Distribution -------- """

class Autoregressive(nn.Module, dist.Distribution):
    """ Represent a generative model that can generate samples and evaluate log probabilities.
        
        Args:
        latt: Lattice
        num_classes: number of classes = group order
        hidden_features: (optional) a list of integers specifying hidden dimensions
    """
    
    def __init__(self, 
                 lattice: Lattice, 
                 num_classes: int, 
                 hidden_features = [],
                 **kwargs):
        super(Autoregressive, self).__init__()
        self.lattice = lattice
        self.graph = self.lattice.causal_graph(**kwargs)
        self.num_classes = num_classes
        features = [self.num_classes] + hidden_features + [self.num_classes]
        self.gcn = GraphConvNet(self.graph, features, **kwargs)
        self.sampler = dist.OneHotCategorical
    
    def sample(self, sample_size: int):
        device = next(self.parameters()).device # determine device
        samples = torch.zeros(sample_size, self.lattice.sites, self.num_classes, device=device) # prepare sample container
        cache = None
        for depth in range(self.graph.max_depth + 1):
            logits, cache = self.gcn(samples, depth, cache)
            select = self.graph.source_depths == depth # select nodes of the depth
            # sample from logits (!this in-place operation stops gradient backprop)
            samples[...,select,:] = self.sampler(logits=logits[...,select,:]).sample()
        return samples
    
    def log_prob(self, samples):
        logits = self.gcn(samples) # forward pass to get logits
        log_prob = torch.sum(samples * F.log_softmax(logits, dim=-1), (-2,-1))
        return log_prob

""" -------- Model Interface -------- """

class PixelGNN(nn.Module):
    """ Autoregressive model based on Graph Neural Network
    
        Args:
        model: a energy model to learn
    """
    def __init__(self, model: Model, **kwargs):
        super(PixelGNN, self).__init__()
        self.model = model
        self.generator = Autoregressive(self.model.lattice, self.model.group.order, **kwargs)
        if isinstance(self.model.lattice, FlatLattice):
            self.configrate = ReshapeTransform(self.model.lattice)
        elif isinstance(self.model.lattice, TreeLattice):
            self.configrate = HaarTransform(self.model.lattice, self.model.group)
        else:
            raise NotImplementedError
        self.categorize = OneHotCategoricalTransform(self.model.group.order)
        self.transform = dist.ComposeTransform([self.categorize, self.configrate])
    
    def extra_repr(self):
        return '(transform): {}'.format(self.transform)

    def sample(self, sample_size: int):
        with torch.no_grad(): # disable gradient to save memory
            z = self.generator.sample(sample_size) 
            x = self.transform(z)
        return x
    
    def energy(self, x):
        return self.model(x)
    
    def log_prob(self, x):
        z = self.transform.inv(x)
        return self.generator.log_prob(z)
    
    def randmix(self, x, mixtures: int):
        x = x.flatten(-self.model.lattice.dimension)
        xs = x[:,self.model.lattice.randperm(mixtures)]
        shape = xs.shape[:-1]+(self.model.lattice.size,)*self.model.lattice.dimension
        return xs.view(shape)
    
    def log_mixed_prob(self, x, mixtures: int):
        xs = self.randmix(x, mixtures)
        log_probs = self.log_prob(xs)
        leading = torch.logsumexp(log_probs, -1) - math.log(mixtures)
        qs = torch.exp(log_probs - log_probs.mean(-1, keepdim=True))
        subleading = qs.var(-1) / qs.mean(-1)**2 
        return leading + subleading / (2 * mixtures)
    
    def loss(self, sample_size: int,
             mixtures: int = None,
             return_statistics: bool = False):
        x = self.sample(sample_size)
        energy = self.energy(x)
        if mixtures is None:
            log_prob = self.log_prob(x)
        else:
            log_prob = self.log_mixed_prob(x, mixtures)
        free = energy + log_prob.detach()
        meanfree = free.mean()
        loss = torch.mean(log_prob * (free - meanfree))
        if return_statistics:
            stdfree = free.std()
            return loss, meanfree.item(), stdfree.item()
        else:
            return loss










