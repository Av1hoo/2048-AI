#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Export functions
__declspec(dllexport) void init_c_game_2048();
__declspec(dllexport) uint64_t bitboard_move(uint64_t bitboard, int action, int* out_score, bool* out_moved);
__declspec(dllexport) int bitboard_count_empty(uint64_t bitboard);
__declspec(dllexport) uint64_t bitboard_spawn(uint64_t bitboard);
__declspec(dllexport) bool bitboard_is_game_over(uint64_t bitboard);
__declspec(dllexport) int bitboard_get_max_tile(uint64_t bitboard);
__declspec(dllexport) void bitboard_to_board_c(uint64_t bitboard, int board[4][4]);
__declspec(dllexport) uint64_t board_to_bitboard_c(const int board[4][4]);
__declspec(dllexport) double advanced_heuristic_c(uint64_t bitboard);

// We’ll store look-up tables as static/global arrays in C:
static uint16_t row_left_table[65536];
static int      row_score_table[65536];

// -----------------------------------------------------------------------------
// 1) Helper to reverse a 16-bit row (4 nibbles)
// -----------------------------------------------------------------------------
static inline uint16_t reverse_row(uint16_t row16) {
    // row16 = (n3 n2 n1 n0) in nibble form
    // we want to swap n0 <-> n3, n1 <-> n2
    uint16_t n0 = (row16 >>  0) & 0xF;
    uint16_t n1 = (row16 >>  4) & 0xF;
    uint16_t n2 = (row16 >>  8) & 0xF;
    uint16_t n3 = (row16 >> 12) & 0xF;
    return (uint16_t)((n0 << 12) | (n1 << 8) | (n2 << 4) | (n3));
}

// -----------------------------------------------------------------------------
// 2) Build row_left_table & row_score_table
//    row_left_table[row16] => the result of sliding that 4-cell row left
//    row_score_table[row16] => the immediate score gained from merges in that move
// -----------------------------------------------------------------------------
void init_tables() {
    // Build row_left_table and row_score_table
    for (uint32_t row_val = 0; row_val < 65536; row_val++) {
        // Extract 4 nibbles:
        uint16_t t0 = (row_val >>  0) & 0xF;
        uint16_t t1 = (row_val >>  4) & 0xF;
        uint16_t t2 = (row_val >>  8) & 0xF;
        uint16_t t3 = (row_val >> 12) & 0xF;
        uint16_t row[4] = { t0, t1, t2, t3 };

        // Slide left logic:
        // 1) Filter out zeroes
        int filtered[4];
        int count = 0;
        for (int i = 0; i < 4; i++) {
            if (row[i] != 0) {
                filtered[count++] = row[i];
            }
        }
        // 2) Merge pairs from left
        int merged[4] = {0,0,0,0};
        int idx_out = 0;
        int score = 0;
        for (int i = 0; i < count; i++) {
            if (i < count - 1 && filtered[i] == filtered[i+1]) {
                // Merge
                int val = filtered[i] + 1; // exponent is incremented
                score += (1 << val);       // 2^val
                merged[idx_out++] = val;
                i++; // skip next
            } else {
                merged[idx_out++] = filtered[i];
            }
        }
        // 3) Pad with zeroes
        while (idx_out < 4) {
            merged[idx_out++] = 0;
        }

        // 4) Reconstruct a 16-bit row
        uint16_t new_val = 0;
        new_val |= (merged[0] & 0xF);
        new_val |= (uint16_t)((merged[1] & 0xF) << 4);
        new_val |= (uint16_t)((merged[2] & 0xF) << 8);
        new_val |= (uint16_t)((merged[3] & 0xF) << 12);

        row_left_table[row_val] = new_val;
        row_score_table[row_val] = score;
    }
}

// -----------------------------------------------------------------------------
// 3) Shift/Slide operations: left, right, up, down
// -----------------------------------------------------------------------------
static inline uint16_t get_row(uint64_t bitboard, int row_idx) {
    // row_idx=0 => lowest 16 bits
    // row_idx=1 => next 16 bits, ...
    return (uint16_t)((bitboard >> (16 * row_idx)) & 0xFFFF);
}

static inline uint64_t set_row(uint64_t bitboard, int row_idx, uint16_t row16) {
    uint64_t mask = (uint64_t)0xFFFF << (16 * row_idx);
    bitboard &= ~mask;
    bitboard |= ((uint64_t)row16 << (16 * row_idx));
    return bitboard;
}

static inline uint64_t transpose(uint64_t bitboard) {
    // We decode the 4x4 exponents, then swap row/col, re-encode.
    // We'll use a small array [16].
    int exps[16];
    for (int r = 0; r < 4; r++) {
        uint16_t row = (uint16_t)((bitboard >> (16*r)) & 0xFFFF);
        for (int c = 0; c < 4; c++) {
            exps[r*4 + c] = (row >> (4*c)) & 0xF;
        }
    }
    // Transpose in exps array
    int trans[16];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            trans[c*4 + r] = exps[r*4 + c];
        }
    }
    // re-encode
    uint64_t b2 = 0;
    for (int r = 0; r < 4; r++) {
        uint16_t rowval = 0;
        for (int c = 0; c < 4; c++) {
            rowval |= (uint16_t)((trans[r*4 + c] & 0xF) << (4*c));
        }
        b2 = set_row(b2, r, rowval);
    }
    return b2;
}

static inline uint64_t shift_left(uint64_t bitboard, int* out_score, bool* out_moved) {
    bool moved = false;
    int total_score = 0;
    uint64_t b2 = bitboard;
    for (int r = 0; r < 4; r++) {
        uint16_t row = get_row(b2, r);
        uint16_t new_row = row_left_table[row];
        if (new_row != row) {
            moved = true;
        }
        total_score += row_score_table[row];
        b2 = set_row(b2, r, new_row);
    }
    *out_score = total_score;
    *out_moved = moved;
    return b2;
}

static inline uint64_t shift_right(uint64_t bitboard, int* out_score, bool* out_moved) {
    bool moved = false;
    int total_score = 0;
    uint64_t b2 = bitboard;
    for (int r = 0; r < 4; r++) {
        uint16_t row = get_row(b2, r);
        uint16_t rev = reverse_row(row);
        uint16_t new_rev = row_left_table[rev];
        if (new_rev != rev) {
            moved = true;
        }
        total_score += row_score_table[rev];
        uint16_t fin = reverse_row(new_rev);
        b2 = set_row(b2, r, fin);
    }
    *out_score = total_score;
    *out_moved = moved;
    return b2;
}

static inline uint64_t shift_up(uint64_t bitboard, int* out_score, bool* out_moved) {
    // transpose -> shift_left -> transpose
    uint64_t t = transpose(bitboard);
    int sc;
    bool mv;
    uint64_t t2 = shift_left(t, &sc, &mv);
    uint64_t b2 = transpose(t2);
    *out_score = sc;
    *out_moved = mv;
    return b2;
}

static inline uint64_t shift_down(uint64_t bitboard, int* out_score, bool* out_moved) {
    // transpose -> shift_right -> transpose
    uint64_t t = transpose(bitboard);
    int sc;
    bool mv;
    uint64_t t2 = shift_right(t, &sc, &mv);
    uint64_t b2 = transpose(t2);
    *out_score = sc;
    *out_moved = mv;
    return b2;
}

// action: 0=Up, 1=Down, 2=Left, 3=Right
// returns new bitboard, out_score, and out_moved
uint64_t bitboard_move(uint64_t bitboard, int action, int* out_score, bool* out_moved) {
    switch (action) {
        case 0:
            return shift_up(bitboard, out_score, out_moved);
        case 1:
            return shift_down(bitboard, out_score, out_moved);
        case 2:
            return shift_left(bitboard, out_score, out_moved);
        case 3:
            return shift_right(bitboard, out_score, out_moved);
        default:
            *out_score = 0;
            *out_moved = false;
            return bitboard;
    }
}

// -----------------------------------------------------------------------------
// 4) Counting empties, spawning, game over, max tile
// -----------------------------------------------------------------------------
int bitboard_count_empty(uint64_t bitboard) {
    int cnt = 0;
    for (int i = 0; i < 16; i++) {
        if ((bitboard & 0xF) == 0) {
            cnt++;
        }
        bitboard >>= 4;
    }
    return cnt;
}

uint64_t bitboard_spawn(uint64_t bitboard) {
    int empties = bitboard_count_empty(bitboard);
    if (empties == 0) return bitboard;

    // pick a random empty index in [0, empties-1]
    // We'll rely on the caller to have seeded random, etc.
    int r = rand() % empties;
    // tile exponent: 1 for '2', 2 for '4' with 10% chance
    int val = ((double)rand()/RAND_MAX < 0.9) ? 1 : 2;

    int seen = 0;
    uint64_t mask = 0xF;
    int shift = 0;
    uint64_t tmp = bitboard;
    while (true) {
        int nib = tmp & 0xF;
        if (nib == 0) {
            if (seen == r) {
                // place val
                uint64_t clearMask = ~( (uint64_t)0xF << shift );
                bitboard &= clearMask;
                bitboard |= ((uint64_t)val << shift);
                break;
            }
            seen++;
        }
        shift += 4;
        tmp >>= 4;
    }
    return bitboard;
}

bool bitboard_is_game_over(uint64_t bitboard) {
    // If there's an empty, not over
    if (bitboard_count_empty(bitboard) > 0) {
        return false;
    }
    // no empty, check merges horizontally
    for (int r = 0; r < 4; r++) {
        uint16_t row = (uint16_t)((bitboard >> (16*r)) & 0xFFFF);
        for (int c = 0; c < 3; c++) {
            int n1 = (row >> (4*c)) & 0xF;
            int n2 = (row >> (4*(c+1))) & 0xF;
            if (n1 == n2) return false;
        }
    }
    // check merges vertically
    uint64_t t = transpose(bitboard);
    for (int r = 0; r < 4; r++) {
        uint16_t row = (uint16_t)((t >> (16*r)) & 0xFFFF);
        for (int c = 0; c < 3; c++) {
            int n1 = (row >> (4*c)) & 0xF;
            int n2 = (row >> (4*(c+1))) & 0xF;
            if (n1 == n2) return false;
        }
    }
    return true;
}

int bitboard_get_max_tile(uint64_t bitboard) {
    int mx = 0;
    for (int i = 0; i < 16; i++) {
        int nib = (int)(bitboard & 0xF);
        if (nib > mx) {
            mx = nib;
        }
        bitboard >>= 4;
    }
    return (1 << mx);
}

// -----------------------------------------------------------------------------
// 5) Convert bitboard <-> 2D array (4x4) of actual tile values
// -----------------------------------------------------------------------------
void bitboard_to_board_c(uint64_t bitboard, int board[4][4]) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int nib = (int)(bitboard & 0xF);
            board[r][c] = (nib > 0) ? (1 << nib) : 0;
            bitboard >>= 4;
        }
    }
}

uint64_t board_to_bitboard_c(const int board[4][4]) {
    uint64_t bitboard = 0;
    int shift = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int val = board[r][c];
            if (val > 0) {
                int exp = (int)(log2(val) + 0.5); // approximate
                bitboard |= ((uint64_t)exp << shift);
            }
            shift += 4;
        }
    }
    return bitboard;
}

// -----------------------------------------------------------------------------
// 6) Heuristics: corner_weighting, monotonicity, empties, max exponent
// -----------------------------------------------------------------------------
static double corner_weighting(uint64_t bitboard) {
    // Example weighting array for exponents
    static const int weights[16] = {
       50,  4,  3,  2,
       10,  5,  1,  1,
        5,  2,  1,  1,
        2,  1,  1,  1
    };
    double value = 0.0;
    for (int i = 0; i < 16; i++) {
        int nib = (int)(bitboard & 0xF);
        bitboard >>= 4;
        value += (double)nib * (double)weights[i];
    }
    return value;
}

static double monotonicity(uint64_t bitboard) {
    // Decode into 4x4 exponent matrix
    int exps[4][4];
    for (int r = 0; r < 4; r++) {
        uint16_t row = (uint16_t)((bitboard >> (16*r)) & 0xFFFF);
        for (int c = 0; c < 4; c++) {
            exps[r][c] = (row >> (4*c)) & 0xF;
        }
    }

    double total = 0.0;
    // Row monotonic
    for (int r = 0; r < 4; r++) {
        double incr = 0.0, decr = 0.0;
        for (int c = 0; c < 3; c++) {
            if (exps[r][c+1] > exps[r][c]) {
                incr += (exps[r][c+1] - exps[r][c]);
            } else {
                decr += (exps[r][c] - exps[r][c+1]);
            }
        }
        total += (incr > decr ? incr : decr);
    }
    // Col monotonic
    for (int c = 0; c < 4; c++) {
        double incr = 0.0, decr = 0.0;
        for (int r = 0; r < 3; r++) {
            if (exps[r+1][c] > exps[r][c]) {
                incr += (exps[r+1][c] - exps[r][c]);
            } else {
                decr += (exps[r][c] - exps[r+1][c]);
            }
        }
        total += (incr > decr ? incr : decr);
    }
    return total;
}

static inline int count_empty(uint64_t bitboard) {
    int cnt = 0;
    for (int i = 0; i < 16; i++) {
        if ((bitboard & 0xF) == 0) cnt++;
        bitboard >>= 4;
    }
    return cnt;
}

double advanced_heuristic_c(uint64_t bitboard) {
    double empties = (double)count_empty(bitboard);
    double corner_val = corner_weighting(bitboard);
    double mono_val = monotonicity(bitboard);

    // max tile
    // We can do a quick pass to find max exponent
    // but note corner_weighting() already consumed bitboard,
    // so let's re-decode or do a second pass from a copy.
    // Let's just decode again or separate calls if you prefer.
    // For simplicity, let's decode again:
    // (Better approach: corner_weighting can read from a local copy)
    // But for clarity we do it again:
    // Actually, let's do a copy before calling corner_weighting:

    // We'll do it the simpler way: let's do a second pass for max exponent.
    // We'll reconstruct bitboard from the function arguments though:
    // Actually let's keep the original argument in a local var:

    // Because corner_weighting() consumed the bits (shifted them out),
    // we need to preserve bitboard first:
    uint64_t copy = bitboard;

    // corner_weighting used shifting, so let's call that first with a local copy:
    double c_val = corner_weighting(copy);

    // Now do monotonicity on a fresh copy:
    copy = bitboard;
    double m_val = monotonicity(copy);

    // empty count again
    copy = bitboard;
    double e_val = (double)count_empty(copy);

    // find max exponent
    copy = bitboard;
    int mx = 0;
    for (int i = 0; i < 16; i++) {
        int nib = (int)(copy & 0xF);
        if (nib > mx) mx = nib;
        copy >>= 4;
    }
    double max_exp = (double)mx;

    // Weighted sum
    double value = 3.0 * c_val
                 + 1.0 * m_val
                 + 2.0 * e_val
                 + 2.0 * max_exp;
    return value;
}

// -----------------------------------------------------------------------------
// 7) Provide a convenient init function to call from Python once
// -----------------------------------------------------------------------------
void init_c_game_2048() {
    init_tables();
    // seed the RNG once if desired:
    srand((unsigned)time(NULL));
}
