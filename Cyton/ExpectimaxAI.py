from functools import lru_cache
import time
from C_funcs import bitboard_move, bitboard_count_empty, bitboard_is_game_over, advanced_heuristic

@lru_cache(maxsize=6000)
class ExpectimaxAI:
    def __init__(self, depth=5, TIME_LIMIT=30):
        self.depth = depth
        self.cache = {}
        # For timing:
        self.start_time = 0
        self.timed_out = False
        self.time_limit = TIME_LIMIT

    def get_move(self, bitboard):
        """
        We start timing here. If we exceed 30 seconds in recursion,
        we stop searching and return whatever best we found so far.
        """
        self.start_time = time.time()
        self.timed_out = False
        best_val = float("-inf")
        best_move = None

        for action in [0, 1, 2, 3]:
            # Check time limit:
            if time.time() - self.start_time > self.time_limit:
                # Timed out; return best so far
                print("Timed out getting move")
                return best_move

            nb, sc, moved = bitboard_move(bitboard, action)
            if not moved:
                continue

            val = sc + self.expectimax_value(nb, self.depth - 1, False)

            if val > best_val:
                best_val = val
                best_move = action

            if self.timed_out:
                # If any deeper call timed out, stop immediately
                break

        return best_move

    def expectimax_value(self, bitboard, depth, is_player):
        # Immediately return if we are over time:
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return 0  # fallback value

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
                    val = sc + self.expectimax_value(nb, depth - 1, False)
                    if val > best_val:
                        best_val = val
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
                val = self.expectimax_value(bitboard, depth - 1, True)
                self.cache[key] = val
                return val

            val_sum = 0.0
            prob2, prob4 = 0.9, 0.1
            shift = 0
            tmp = bitboard
            for _ in range(16):
                if time.time() - self.start_time > self.time_limit:
                    self.timed_out = True
                    break
                nib = (tmp & 0xF)
                if nib == 0:
                    b2 = (bitboard & ~(0xF << shift)) | (1 << shift)
                    v2 = self.expectimax_value(b2, depth - 1, True)
                    b4 = (bitboard & ~(0xF << shift)) | (2 << shift)
                    v4 = self.expectimax_value(b4, depth - 1, True)
                    val_sum += (prob2 * v2 + prob4 * v4)
                tmp >>= 4
                shift += 4

            # If we timed out halfway, no sense dividing partial sums by empty,
            # but let's still do it so we can return a partial expectation:
            val = val_sum / empty if empty > 0 else 0
            self.cache[key] = val
            return val