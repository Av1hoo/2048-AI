import torch
from C_funcs import bitboard_move, bitboard_to_explist
from torch import nn


class DqnAI:
    class DQNModel(nn.Module):
        def __init__(self, state_dim=16, hidden_dim=256, action_dim=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        def forward(self, x):
            return self.net(x)
        
    def __init__(self, DQN_MODEL_PATH="model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn_model = self.DQNModel()
        try:
            ckpt = torch.load(DQN_MODEL_PATH, map_location=self.device)
            dqn_model.load_state_dict(ckpt, strict=False)
            dqn_model.to(self.device)
            dqn_model.eval()
        except:
            dqn_model = None

    def get_move(self, bitboard):
        if not self.dqn_model:
            return None
        exps = bitboard_to_explist(bitboard)
        st = torch.tensor(exps, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.dqn_model(st).cpu().numpy()[0]
        order = sorted(range(4), key=lambda i: q[i], reverse=True)
        for a in order:
            nb, _, moved = bitboard_move(bitboard, a)
            if moved:
                return a
        return None

