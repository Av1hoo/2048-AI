import torch
from CytonFiles.C_funcs import bitboard_move
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
            ckpt = torch.load(DQN_MODEL_PATH, map_location=self.device, weights_only=True)
            self.dqn_model.load_state_dict(ckpt, strict=False)
            self.dqn_model.to(self.device)
            self.dqn_model.eval()
            print(f"Model loaded successfully and is on device: {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.dqn_model = None

    def bitboard_to_explist(self, bb):
        arr = []
        for _ in range(16):
            arr.append(bb & 0xF)
            bb >>= 4
        return arr

    def get_move(self, bitboard):
        if not self.dqn_model:
            return None
        
        exps = self.bitboard_to_explist(bitboard)
        # Ensure input tensor is on correct device
        st = torch.tensor(exps, dtype=torch.float32).unsqueeze(0)
        
        # Move input to same device as model
        st = st.to(self.device)
        
        with torch.no_grad():
            # Model and input now on same device
            q = self.dqn_model(st)
            # Move output to CPU for numpy conversion
            q = q.cpu().numpy()[0]
        
        order = sorted(range(4), key=lambda i: q[i], reverse=True)
        for a in order:
            nb, _, moved = bitboard_move(bitboard, a)
            if moved:
                return a
        return None
