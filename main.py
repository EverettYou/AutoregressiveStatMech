import argparse
import torch
import torch.optim as optim
import pixel_gnn.model as model

# set device
parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
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
                model.Lattice(4, 2)), 
            hidden_features = [4, 4]).to(model.device)
optimizer = optim.Adam(hpGNN.parameters(), lr=0.02)

# training
batch_size = 100
train_loss = 0.
free_energy = 0.
echo = 100
for epoch in range(2000):
    x = model.sample(batch_size)
    log_prob = model.log_prob(x)
    energy = model.energy(x)
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