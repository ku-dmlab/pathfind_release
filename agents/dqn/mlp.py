from torch import Tensor, nn
from torch.nn import functional as F

class QNet(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_hidden: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.fc_in = nn.Linear(state_size, hidden_dim)
        self.fc_hiddens = nn.ModuleList()
        
        for _ in range(num_hidden):
            self.fc_hiddens.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_hiddens.append(nn.ReLU())
        self.fc_out = nn.Linear(hidden_dim, action_size)
        
    def forward(self, x:Tensor) -> Tensor:
        x = F.relu(self.fc_in(x))
        for layer in self.fc_hiddens:
            x = layer(x)
        return self.fc_out(x)