# replay_2048.py

import struct
import math

#####################
# Utility
#####################
def decode_bitboard_to_2d(bitboard):
    """Returns a 4x4 array of tile-values (0,2,4,8,...) from the 64-bit bitboard."""
    board = [[0]*4 for _ in range(4)]
    for i in range(16):
        exp = (bitboard >> (4*i)) & 0xF
        val = (0 if exp==0 else (1 << exp))  # 2^exp
        r, c = divmod(i, 4)
        board[r][c] = val
    return board

def read_all_games(filename):
    """Return a list of games. Each game is a list of (bitboard, move)."""
    all_games = []
    with open(filename, "rb") as f:
        raw = f.read(8)
        (numGames,) = struct.unpack("<Q", raw)
        for _ in range(numGames):
            raw = f.read(4)
            (steps,) = struct.unpack("<I", raw)
            moves = []
            for __ in range(steps):
                chunk = f.read(9)
                bitboard, mv = struct.unpack("<Qb", chunk)
                move = mv & 0xFF
                moves.append((bitboard, move))
            all_games.append(moves)
    return all_games

#####################
# Command-line replay
#####################
import struct
import math
import os
import msvcrt
import time

def get_color(value):
    """Returns ANSI color code based on tile value"""
    colors = {
        0: '\033[90m',      # gray
        2: '\033[97m',      # white
        4: '\033[93m',      # yellow
        8: '\033[33m',      # orange
        16: '\033[91m',     # red
        32: '\033[31m',     # dark red
        64: '\033[95m',     # magenta
        128: '\033[94m',    # blue
        256: '\033[96m',    # cyan
        512: '\033[92m',    # green
        1024: '\033[32m',   # dark green
        2048: '\033[35m',   # purple
    }
    return colors.get(value, '\033[0m')

def print_board(board_2d):
    """Enhanced board printing with colors and box drawing"""
    os.system('cls')  # Clear screen for Windows
    print("╔═══════╦═══════╦═══════╦═══════╗")
    for i, row in enumerate(board_2d):
        print("║", end="")
        for val in row:
            color = get_color(val)
            if val == 0:
                print(f"{color}   ·   \033[0m║", end="")
            else:
                print(f"{color}{val:^7}\033[0m║", end="")
        print()
        if i < 3:
            print("╠═══════╬═══════╬═══════╬═══════╣")
    print("╚═══════╩═══════╩═══════╩═══════╝")

def get_game_moves_count(game):
    """Returns the number of moves in a game"""
    return len(game)

def main():
    filename = "games.bin"
    all_games = read_all_games(filename)
    print(f"Loaded {len(all_games)} games.")
    
    # Create list of tuples (game, original_index, moves_count)
    indexed_games = [(game, idx, get_game_moves_count(game)) 
                    for idx, game in enumerate(all_games)]
    
    # Sort by moves count in descending order
    indexed_games.sort(key=lambda x: x[2], reverse=True)
    
    current_game_idx = 0
    current_step_idx = 0
    move_to_str = {0: "Right", 1: "Left", 2: "Down", 3: "Up"}
    
    while True:
        game, original_idx, moves_count = indexed_games[current_game_idx]
        bitboard, move = game[current_step_idx]
        b2d = decode_bitboard_to_2d(bitboard)
        
        print_board(b2d)
        print(f"\nGame {current_game_idx+1}/{len(indexed_games)} (Original #{original_idx+1})")
        print(f"Move {current_step_idx+1}/{moves_count} | Total moves: {moves_count}")
        print(f"Next move: {move_to_str[move]}")
        print("\nControls: [N]ext [P]rev [G]ame+ [B]ack-game [Q]uit")

        while not msvcrt.kbhit():
            time.sleep(0.1)
        
        key = msvcrt.getch().decode().lower()
        
        if key == 'n':
            if current_step_idx < len(game)-1:
                current_step_idx += 1
        elif key == 'p':
            if current_step_idx > 0:
                current_step_idx -= 1
        elif key == 'g':
            if current_game_idx < len(indexed_games)-1:
                current_game_idx += 1
                current_step_idx = 0
        elif key == 'b':
            if current_game_idx > 0:
                current_game_idx -= 1
                current_step_idx = 0
        elif key == 'q':
            break

if __name__ == "__main__":
    main()