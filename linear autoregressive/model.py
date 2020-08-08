import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

class AutoregressiveLinear(nn.Linear):
    """ Applies a lienar transformation to the incoming data, 
        with the weight matrix masked to the lower-triangle. 
        
        Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        diagonal: the diagonal to trucate to"""
    
    def __init__(self, units, in_features, out_features, bias=True, diagonal=0):
        super(AutoregressiveLinear, self).__init__(units*in_features, units*out_features, bias)
        self.units = units
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal = diagonal
        self.mask = torch.tensordot(torch.tril(torch.ones(units, units), diagonal), torch.ones(out_features, in_features), dims=0).transpose(1,2).reshape(units*out_features, units*in_features)

    
    def extra_repr(self):
        return 'unites={}, in_features={}, out_features={}, bias={}, diagonal={}'.format(self.units, self.in_features, self.out_features, not self.bias is None, self.diagonal)
    
    # overwrite forward pass
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
    
    def forward_at(self, input, i):
        # pick out the weight block that is active
        active_weight = self.weight.narrow(0, i*self.out_features, self.out_features) # narrow out the rows
        active_weight = active_weight.narrow(1, 0, (i + 1 + self.diagonal)*self.in_features) # narrow out the columns
        # pick out the input block that is active
        active_input = input.narrow(-1, 0, (i + 1 + self.diagonal)*self.in_features)
        # transform active input by active weight
        output = active_input.matmul(active_weight.t())
        if self.bias is not None: # if bias exists, add it
            output += self.bias.narrow(0, i*self.out_features, self.out_features)
        return output

class AutoregressiveModel(nn.Module):
    """ Represent a generative model that can generate samples and provide log probability evaluations.
        
        Args:
        units: number of units in the model
        features: a list of feature dimensions from the input layer to the output layer
        nonlinearity: activation function to use """
    
    def __init__(self, units, features, nonlinearity='Tanh'):
        super(AutoregressiveModel, self).__init__()
        self.units = units
        self.features = features
        if features[0] != features[-1]:
            raise ValueError('In features {}, the first and last feature dimensions must be equal.'.format(features))
        self.layers = nn.ModuleList()
        for l in range(len(features)-1):
            if l == 0: # first autoregressive linear layer must have diagonal=-1
                self.layers.append(AutoregressiveLinear(units, features[0], features[1], diagonal = -1))
            else: # remaining autoregressive linear layers have diagonal=0 (by default)
                self.layers.append(getattr(nn, nonlinearity)())
                self.layers.append(AutoregressiveLinear(units, features[l], features[l+1]))
    
    def extra_repr(self):
        return '(units): {}\n(features): {}'.format(self.units, self.features) + super(AutoregressiveModel, self).extra_repr()
    
    def forward(self, input):
        logits = input # logits as a workspace, initialized to input
        for layer in self.layers: # apply layers
            logits = layer(logits)
        return logits # logits output
    
    def log_prob(self, input):
        logits = self(input).view(-1, self.units, self.features[-1]) # forward pass to get logits
        input = input.view(-1, self.units, self.features[0])
        return torch.sum(F.softmax(logits, dim=-1).log() * input, (-2,-1))
        
    def _xsample(self, batch_size, tau, hard):
        # create a list to host layer-wise outputs
        record = [torch.empty(batch_size, 0) for _ in self.features]
        # autoregressive batch sampling
        for i in range(self.units):
            for l in range(len(self.features)-1):
                if l==0: # first linear layer
                    output = self.layers[0].forward_at(record[0], i)
                else: # remaining layers
                    output = self.layers[2*l-1](output) # element-wise layer
                    record[l] = torch.cat([record[l], output], dim=-1) # concatenate output to record
                    output = self.layers[2*l].forward_at(record[l], i)
            # record[-1] = torch.cat([record[-1],  output], dim=-1) # for debug purpose
            # the last output hosts logits, sample by Gumbel softmax 
            sample = F.gumbel_softmax(output, tau, hard)
            record[0] = torch.cat([record[0], sample], dim=-1) # concatenate sample to record
        return record
    
    def rsample(self, batch_size=1, tau=None, hard=False):
        if tau is None: # if temperature not given
            tau = 1/(self.features[-1]-1) # set by the out feature dimension
        return self._xsample(batch_size, tau, hard)[0]
    
    def sample(self, batch_size=1, tau=None, hard=False):
        with torch.no_grad(): # no gradient for sample generation
            return self.rsample(batch_size, tau, hard)

class StatMechSystem(nn.Module):
    """ Provide evaluation for the transfer weight and its marginalization.
        
        Args:
        units: number of units in the model (system size)
        bond_weight: bond weight matrix"""
    
    def __init__(self, units, bond_weight):
        super(StatMechSystem, self).__init__()
        self.units = units
        self.W = bond_weight
        self.w = bond_weight.sum(0)
        self.states = len(bond_weight)
        
    def forward(self, *xs):
        # receive configurations and view in tensor form
        x = None
        if len(xs) == 1:
            xp = xs[0].view(1, -1, self.units, self.states)
        elif len(xs) == 2:
            x = xs[0].view(-1, 1, self.units, self.states)
            xp = xs[1].view(1, -1, self.units, self.states)
        else:
            raise ValueError('Expect 1 or 2 arguments. Get {} arguments.'.format(len(xs)))
        # compute the horizontal product
        Th = torch.prod(torch.sum(xp.matmul(self.W) * xp.roll(1, -2), -1), -1)
        # compute the vertical product
        if x is None:
            Tv = torch.prod(xp.matmul(self.w), -1)
        else:
            Tv = torch.prod(torch.sum(x.matmul(self.W) * xp, -1), -1)
        return Th * Tv