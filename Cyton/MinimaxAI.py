import time
from C_funcs import bitboard_move, bitboard_count_empty, bitboard_is_game_over, advanced_heuristic

class MinimaxAI:
    def __init__(self, depth=4, TIME_LIMIT=30):
        self.depth = depth
        self.cache = {}
        # For timing:
        self.start_time = 0
        self.timed_out = False
        self.time_limit = TIME_LIMIT

    def get_move(self, bitboard):
        self.start_time = time.time()
        self.timed_out = False
        best_val = float("-inf")
        best_move = None

        for a in [0, 1, 2, 3]:
            if time.time() - self.start_time > self.time_limit:
                return best_move

            nb, sc, moved = bitboard_move(bitboard, a)
            if moved:
                val = sc + self.minimax_value(nb, self.depth - 1, False)
                if val > best_val:
                    best_val = val
                    best_move = a
            if self.timed_out:
                break

        return best_move

    def minimax_value(self, bitboard, depth, is_player):
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return 0

        key = (bitboard, depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth <= 0 or bitboard_is_game_over(bitboard):
            val = advanced_heuristic(bitboard)
            self.cache[key] = val
            return val

        if is_player:
            best_val = float("-inf")
            anymove = False
            for a in [0, 1, 2, 3]:
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nb, sc, moved = bitboard_move(bitboard, a)
                if moved:
                    anymove = True
                    v = sc + self.minimax_value(nb, depth - 1, False)
                    if v > best_val:
                        best_val = v
                if self.timed_out:
                    break
            if not anymove:
                val = advanced_heuristic(bitboard)
                self.cache[key] = val
                return val
            self.cache[key] = best_val
            return best_val
        else:
            empty = bitboard_count_empty(bitboard)
            if empty == 0:
                val = self.minimax_value(bitboard, depth - 1, True)
                self.cache[key] = val
                return val
            best_val = float("inf")
            shift = 0
            tmp = bitboard
            for _ in range(16):
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nib = tmp & 0xF
                if nib == 0:
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)
                    v2 = self.minimax_value(b2, depth - 1, True)
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)
                    v4 = self.minimax_value(b4, depth - 1, True)
                    worst = v2 if v2 < v4 else v4
                    if worst < best_val:
                        best_val = worst
                tmp >>= 4
                shift += 4
            self.cache[key] = best_val
            return best_val