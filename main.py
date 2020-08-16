import argparse
import torch
import torch.optim as optim
import pixel_gnn.model as model

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', 
                    action='store_true', 
                    help='Disable CUDA')
parser.add_argument('--hidden_features', 
                    type=int,
                    default=8,
                    help='number of hidden features')
parser.add_argument('--depth', 
                    type=int,
                    default=6,
                    help='depth of RNN')
parser.add_argument('--nonlinearity', 
                    type=str,
                    default='Tanh',
                    help='nonlinearity to use')
parser.add_argument('--bias', 
                    type=bool,
                    default=True,
                    help='bias to learn')
parser.add_argument('--lattice_size', 
                    type=int,
                    default=4,
                    help='lattice size')
parser.add_argument('--batch_size', 
                    type=int,
                    default=5000,
                    help='batch size')
parser.add_argument('--learning_rate', 
                    type=int,
                    default=0.02,
                    help='batch size')

# set device
args = parser.parse_args()
if not args.disable_cuda and torch.cuda.is_available():
    model.device = torch.device('cuda')
else:
    model.device = torch.device('cpu')

# setup model
H = lambda J: -J*(model.TwoBody(torch.tensor([1.,-1.]), (1,0))
                + model.TwoBody(torch.tensor([1.,-1.]), (0,1)))
hpGNN = model.HolographicPixelGNN(
            model.Energy(
                H(0.440686793), # Ising critical point
                model.SymmetricGroup(2), 
                model.Lattice(args.lattice_size, 2)), 
            args.hidden_features, args.depth,
            args.nonlinearity, args.bias).to(model.device)
optimizer = optim.Adam(hpGNN.parameters(), lr=args.learning_rate)

# training
batch_size = args.batch_size
train_loss = 0.
free_energy = 0.
echo = 100
for epoch in range(2000):
    x = hpGNN.sample(batch_size)
    log_prob = hpGNN.log_prob(x)
    energy = hpGNN.energy(x)
    free = energy + log_prob.detach()
    meanfree = free.mean()
    loss = torch.sum(log_prob * (free - meanfree))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    free_energy += meanfree.item()
    if (epoch+1)%echo == 0:
        print('{:5} loss: {:8.4f}, free energy: {:8.4f}'.format(epoch+1, train_loss/echo, free_energy/echo))
        train_loss = 0.
        free_energy = 0.