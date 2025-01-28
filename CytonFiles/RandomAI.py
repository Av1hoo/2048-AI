import random
from CytonFiles.C_funcs import bitboard_move

class RandomAI:
    def get_move(self, bitboard):
        random.shuffle([0, 1, 2, 3])
        for a in [0, 1, 2, 3]:
            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                return a
        return None