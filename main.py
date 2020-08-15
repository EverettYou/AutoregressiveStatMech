import argparse
import torch
import torch.optim as optim
import pixel_gnn.model as model

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
if not args.disable_cuda and torch.cuda.is_available():
    model.device = torch.device('cuda')
else:
    model.device = torch.device('cpu')


H = lambda J: -J*(model.TwoBody(torch.tensor([1.,-1.]), (1,0))
                + model.TwoBody(torch.tensor([1.,-1.]), (0,1)))
hpGNN = model.HolographicPixelGNN(
            model.Energy(
                H(0.440686793), # Ising critical point
                model.SymmetricGroup(2), 
                model.Lattice(4, 2)), 
            hidden_features = [4, 4]).to(model.device)
optimizer = optim.Adam(hpGNN.parameters(), lr=0.02)

x = hpGNN.sample(3)
print(x)