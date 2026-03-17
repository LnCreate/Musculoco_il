from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
import torch.nn as nn
import torch.optim as optim
import torch

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fc = nn.Linear(2,2)
    def forward(self, x): return self.fc(x)

r = Regressor(TorchApproximator, input_shape=(2,), output_shape=(2,), network=Net, optimizer={'class':optim.Adam, 'params':{'lr':1e-3}})
print(hasattr(r, 'model') and hasattr(r.model, '_optimizer'))
