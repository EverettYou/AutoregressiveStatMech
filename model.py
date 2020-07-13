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
    
    def __init__(self, in_features, out_features, bias=True, diagonal=0):
        super(AutoregressiveLinear, self).__init__(in_features, out_features, bias)
        self.diagonal = diagonal
    
    def extra_repr(self):
        return super(AutoregressiveLinear, self).extra_repr() + ', diagonal={}'.format(self.diagonal)
    
    # overwrite forward pass
    def forward(self, input):
        return F.linear(input, torch.tril(self.weight, self.diagonal), self.bias)
    
    def forward_at(self, input, i):
        output = input.matmul(torch.tril(self.weight, self.diagonal).narrow(0, i, 1).t())
        if self.bias is not None:
            output += self.bias.narrow(0, i, 1)
        return output.squeeze()

class AutoregressiveModel(nn.Module):
    """ Represent a generative model that can generate samples and provide log probability evaluations.
        
        Args:
        features: size of each sample
        depth: depth of the neural network (in number of linear layers) (default=1)
        nonlinearity: activation function to use (default='ReLU') """
    
    def __init__(self, features, depth=1, nonlinearity='ReLU'):
        super(AutoregressiveModel, self).__init__()
        self.features = features # number of features
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0: # first autoregressive linear layer must have diagonal=-1
                self.layers.append(AutoregressiveLinear(self.features, self.features, diagonal = -1))
            else: # remaining autoregressive linear layers have diagonal=0 (by default)
                self.layers.append(AutoregressiveLinear(self.features, self.features))
            if i == depth-1: # the last layer must be Sigmoid
                self.layers.append(nn.Sigmoid())
            else: # other layers use the specified nonlinearity
                self.layers.append(getattr(nn, nonlinearity)())
    
    def extra_repr(self):
        return '(features): {}'.format(self.features) + super(AutoregressiveModel, self).extra_repr()
    
    def forward(self, input):
        prob = input # prob as a workspace, initialized to input
        for layer in self.layers: # apply layers
            prob = layer(prob)
        return prob # prob holds predicted Beroulli probability parameters
    
    def log_prob(self, input):
        prob = self(input) # forward pass to get Beroulli probability parameters
        return torch.sum(dist.Bernoulli(prob).log_prob(input), axis=-1)
    
    def sample(self, batch_size=1):
        with torch.no_grad(): # no gradient for sample generation
            # create a record to host layerwise outputs
            record = torch.zeros(len(self.layers)+1, batch_size, self.features)
            # autoregressive batch sampling
            for i in range(self.features):
                for l, layer in enumerate(self.layers):
                    if isinstance(layer, AutoregressiveLinear): # linear layer
                        record[l+1, :, i] = layer.forward_at(record[l], i)
                    else: # elementwise layer
                        record[l+1, :, i] = layer(record[l, :, i])
                record[0, :, i] = dist.Bernoulli(record[-1, :, i]).sample()
        return record[0]
    
    def rsample(self, batch_size=1, tau=1, hard=False):
        # create a record to host layerwise outputs
        record = torch.zeros(len(self.layers)+1, batch_size, self.features)
        # autoregressive batch sampling
        for i in range(self.features):
            for l, layer in enumerate(self.layers):
                if isinstance(layer, AutoregressiveLinear): # linear layer
                    record[l+1, :, i] = layer.forward_at(record[l], i)
                else: # elementwise layer
                    record[l+1, :, i] = layer(record[l, :, i])
            record[0, :, i] = dist.Bernoulli(record[-1, :, i]).sample()
            #prob = record[-1, :, i]
            #logits = torch.stack([prob, 1.-prob], dim=-1).log()
            #print(F.gumbel_softmax(logits, tau, hard))
            #record[0, :, i] = F.gumbel_softmax(logits, tau, hard)
        return record[0]