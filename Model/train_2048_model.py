# improved/train_2048_model.py

import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import math

##############################
# Read the binary file iteratively
##############################
def bitboard_move_generator(filename):
    """
    Generator that yields (bitboard, move) tuples one at a time.
    """
    with open(filename, "rb") as f:
        # 1) read numGames (8 bytes)
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("File too short to contain number of games.")
        (numGames,) = struct.unpack("<Q", raw)
        print(f"Number of games: {numGames}")
        
        for game_idx in range(numGames):
            # Read number of steps for this game
            raw_steps = f.read(4)
            if len(raw_steps) < 4:
                raise ValueError(f"File ended unexpectedly while reading steps for game {game_idx}.")
            (steps,) = struct.unpack("<I", raw_steps)
            
            for step_idx in range(steps):
                # Read a single step: (bitboard, move)
                step_size = 9  # 8 bytes for bitboard + 1 byte for move
                step_data = f.read(step_size)
                if len(step_data) < step_size:
                    raise ValueError(f"File ended unexpectedly while reading step {step_idx} of game {game_idx}.")
                bitboard, mv = struct.unpack("<Qb", step_data)
                move = mv & 0xFF
                yield (bitboard, move)

##############################
# PyTorch IterableDataset
##############################
class BitboardIterableDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        return bitboard_move_generator(self.filename)

##############################
# Simple NextMoveModel (unchanged)
##############################
class NextMoveModel(nn.Module):
    """
    We'll assume tile exponent in [0..15] => vocab_size=16
    Each board is 16 cells => input shape (batch, 16).
    We'll embed each cell, then feed into MLP that outputs
    4 possible moves.
    """
    def __init__(self, vocab_size=16, embed_dim=16, hidden_dim=64, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim*16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 16) => each cell is in [0..15]
        x = self.embedding(x)           # => (batch_size, 16, embed_dim)
        x = x.view(x.size(0), -1)       # => (batch_size, 16*embed_dim)
        x = self.relu(self.fc1(x))      # => (batch_size, hidden_dim)
        x = self.fc2(x)                 # => (batch_size, output_dim)
        return x

##############################
# Custom Collate Function
##############################
def custom_collate(batch):
    """
    Custom collate function to handle bitboards as uint64 and moves as long.
    """
    bitboards = []
    moves = []
    for item in batch:
        bitboard, move = item
        bitboards.append(bitboard)
        moves.append(move)
    # Convert lists to tensors with appropriate dtypes
    bitboards = torch.tensor(bitboards, dtype=torch.uint64)
    moves = torch.tensor(moves, dtype=torch.long)
    return bitboards, moves

##############################
# Decoding Function
##############################
def decode_bitboards(bitboards):
    # Convert uint64 bitboards to proper tensor format
    boards = bitboards.to(torch.int64)
    exponents = []
    
    # Extract 4-bit values using masks and shifts
    for shift in range(0, 64, 4):
        mask = (boards >> shift) & 0xF
        exponents.append(mask)
    
    # Stack and reshape to get board representation
    exponents = torch.stack(exponents, dim=1)  # (batch_size, 16)
    return exponents

##############################
# Training Function using IterableDataset
##############################
def train_model(bin_file="games.bin", epochs=5, batch_size=64, lr=1e-3):
    dataset = BitboardIterableDataset(bin_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # Shuffling is not supported with IterableDataset
        num_workers=0,           # Set to 0 to prevent multiple workers reading the same file
        collate_fn=custom_collate
    )
    # Note: If you need to speed up data loading, consider implementing
    # a proper multi-worker approach where each worker reads a distinct file shard.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NextMoveModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for boards, moves in dataloader:
            # Decode bitboards to exponents
            exponents = decode_bitboards(boards).to(device)  # (batch_size, 16)
            moves = moves.to(device)                           # (batch_size,)

            optimizer.zero_grad()
            outputs = model(exponents)                         # (batch_size, 4)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            if batch_count % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss = {avg_loss:.4f}")

    return model

##############################
# Main Execution
##############################
if __name__ == "__main__":
    model = train_model("games.bin", epochs=5, batch_size=64, lr=1e-3)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")
