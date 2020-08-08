import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math
class HeapLinear(nn.Module):
    """Applies a heap linear transformation to the incoming data (assuming binary heap)
    
    Args:
        nodes: number nodes in the heap tree (better be 2^n-1) 
        in_features: size of input features at each heap node
        out_features: size of output features at each heap node
        bias: whether to learn an additive bias
        minheap: minimum heap step to start, must be non-negative int (default = 0)
    """
    def __init__(self, nodes: int, in_features: int, out_features: int, bias: bool = True, minheap: int = 0):
        super(HeapLinear, self).__init__()
        self.nodes = nodes
        self.depth = (nodes-1).bit_length()+1 # fast log2 ceiling
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.minheap = minheap
        self.linears = nn.ModuleDict()
        for k in range(minheap, self.depth):
            self.linears[str(k)] = nn.Linear(in_features, out_features, bias)
    
    def forward(self, input: torch.Tensor):
        output = torch.zeros(input.size()[:-1]+(self.out_features,))
        for l in range(self.depth):
            in_node0 = math.floor(2**(l-1))
            in_node1 = 2**l
            block_input = input[..., in_node0:in_node1, :]
            for k, linear in self.linears.items():
                k = int(k)
                if in_node0 == 0:
                    heap_factor = math.ceil(2**(k-1))
                    out_node0 = math.floor(2**(k-1))
                    out_node1 = out_node0 + heap_factor
                else:
                    heap_factor = 2**k
                    out_node0 = in_node0 * heap_factor
                    out_node1 = in_node1 * heap_factor
                if out_node1 <= self.nodes:
                    block_output = linear(block_input).repeat_interleave(heap_factor, dim=-2)
                    output[..., out_node0:out_node1, :] += block_output
        return output     
    
    def forward_from(self, input: torch.Tensor, in_node: int):
        output = torch.zeros(input.size()[:-2]+(self.nodes, self.out_features,))
        in_node_dim = input.size(-2)
        if in_node_dim == self.nodes:
            input = input[..., [in_node], :]
        elif in_node_dim != 1:
            raise ValueError('The node dimension must be either {} or 1, get {}.'.format(self.nodes, in_node_dim))
        for k, linear in self.linears.items():
            k = int(k)
            if in_node == 0:
                heap_factor = math.ceil(2**(k-1))
                out_node0 = math.floor(2**(k-1))
                out_node1 = out_node0 + heap_factor
            else:
                heap_factor = 2**k
                out_node0 = in_node * heap_factor
                out_node1 = (in_node + 1) * heap_factor
            if out_node1 <= self.nodes:
                output[..., out_node0:out_node1, :] += linear(input)
        return output
    
    
class AutoregressiveModel(nn.Module, dist.Distribution):
    """ Represent a generative model that can generate samples and evaluate log probabilities.
        
        Args:
        units: number of units in the model
        features: a list of feature dimensions from the input layer to the output layer
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    
    def __init__(self, units: int, features, nonlinearity: str = 'ReLU', bias: bool = True):
        super(AutoregressiveModel, self).__init__()
        self.units = units
        self.features = features
        if features[0] != features[-1]:
            raise ValueError('In features {}, the first and last feature dimensions must be equal.'.format(features))
        self.layers = nn.ModuleList()
        for l in range(len(features)-1):
            if l == 0: # first heap linear layer must have minheap=1
                self.layers.append(HeapLinear(units, features[0], features[1], bias, minheap = 1))
            else: # remaining heap linear layers have minheap=0 (by default)
                self.layers.append(getattr(nn, nonlinearity)())
                self.layers.append(HeapLinear(units, features[l], features[l+1], bias))
        dist.Distribution.__init__(self, event_shape=torch.Size([units, features[0]]))
        self.has_rsample = True
    
    def extra_repr(self):
        return '(units): {}\n(features): {}'.format(self.units, self.features) + super(AutoregressiveModel, self).extra_repr()
    
    def forward(self, input):
        logits = input # logits as a workspace, initialized to input
        for layer in self.layers: # apply layers
            logits = layer(logits)
        return logits # logits output
    
    def log_prob(self, value):
        logits = self(value) # forward pass to get logits
        return torch.sum(value * F.log_softmax(logits, dim=-1), (-2,-1))
        
    def _sample(self, batch_size: int, sampler = None):
        if sampler is None: # use default sampler
            sampler = self.sampler
        # create a list of tensors to cache layer-wise outputs
        cache = [torch.zeros(batch_size, self.units, feature) for feature in self.features]
        # autoregressive batch sampling
        for i in range(self.units):
            for l in range(len(self.features)-1):
                if l==0: # first linear layer
                    if i > 0:
                        cache[1] += self.layers[0].forward_from(cache[0], i - 1) # heap linear
                else: # remaining layers
                    activation = self.layers[2*l-1](cache[l][..., [i], :]) # element-wise
                    cache[l + 1] += self.layers[2*l].forward_from(activation, i) # heap linear
            # the last record hosts logits 
            cache[0][..., i, :] = sampler(cache[-1][..., i, :])
        return cache # cache[0] hosts the sample
        
    def sampler(self, logits, dim=-1): # simplified from F.gumbel_softmax
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels += logits
        index = gumbels.max(dim, keepdim=True)[1]
        return torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    
    def sample(self, batch_size=1):
        with torch.no_grad():
            cache = self._sample(batch_size)
        return cache[0]
    
    def rsample(self, batch_size=1, tau=None, hard=False):
        if tau is None: # if temperature not given
            tau = 1/(self.features[-1]-1) # set by the out feature dimension
        cache = self._sample(batch_size, lambda x: F.gumbel_softmax(x, tau, hard))
        return cache[0]