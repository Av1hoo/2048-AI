import ctypes
###############################################################################
# 1) Load the C Library via ctypes
###############################################################################
libname = "CytonFiles\C\c_game2048.dll"
c2048 = ctypes.CDLL(libname)

c2048.init_c_game_2048.argtypes = []
c2048.init_c_game_2048.restype = None

c2048.bitboard_move.argtypes = [ctypes.c_uint64, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_bool)]
c2048.bitboard_move.restype = ctypes.c_uint64

c2048.bitboard_count_empty.argtypes = [ctypes.c_uint64]
c2048.bitboard_count_empty.restype = ctypes.c_int

c2048.bitboard_spawn.argtypes = [ctypes.c_uint64]
c2048.bitboard_spawn.restype = ctypes.c_uint64

c2048.bitboard_is_game_over.argtypes = [ctypes.c_uint64]
c2048.bitboard_is_game_over.restype = ctypes.c_bool

c2048.bitboard_get_max_tile.argtypes = [ctypes.c_uint64]
c2048.bitboard_get_max_tile.restype = ctypes.c_int

Int4Array = ctypes.c_int * 4
Board4x4 = Int4Array * 4
c2048.bitboard_to_board_c.argtypes = [ctypes.c_uint64, Board4x4]
c2048.bitboard_to_board_c.restype = None

c2048.board_to_bitboard_c.argtypes = [Board4x4]
c2048.board_to_bitboard_c.restype = ctypes.c_uint64

c2048.advanced_heuristic_c.argtypes = [ctypes.c_uint64]
c2048.advanced_heuristic_c.restype = ctypes.c_double

def init_c_game():
    c2048.init_c_game_2048()

def bitboard_move(bitboard, action):
    out_score = ctypes.c_int(0)
    out_moved = ctypes.c_bool(False)
    nb = c2048.bitboard_move(ctypes.c_uint64(bitboard),
                             ctypes.c_int(action),
                             ctypes.byref(out_score),
                             ctypes.byref(out_moved))
    return nb, out_score.value, out_moved.value

def bitboard_count_empty(bitboard):
    return c2048.bitboard_count_empty(ctypes.c_uint64(bitboard))

def bitboard_spawn(bitboard):
    return c2048.bitboard_spawn(ctypes.c_uint64(bitboard))

def bitboard_is_game_over(bitboard):
    return c2048.bitboard_is_game_over(ctypes.c_uint64(bitboard))

def bitboard_get_max_tile(bitboard):
    return c2048.bitboard_get_max_tile(ctypes.c_uint64(bitboard))

def bitboard_to_board(bitboard):
    b = Board4x4()
    c2048.bitboard_to_board_c(ctypes.c_uint64(bitboard), b)
    pyboard = []
    for r in range(4):
        row = []
        for c in range(4):
            row.append(b[r][c])
        pyboard.append(row)
    return pyboard

def board_to_bitboard(mat):
    b = Board4x4()
    for r in range(4):
        for c in range(4):
            b[r][c] = mat[r][c]
    return c2048.board_to_bitboard_c(b)

def advanced_heuristic(bitboard):
    return c2048.advanced_heuristic_c(ctypes.c_uint64(bitboard))

def bitboard_spawn_func(bitboard):
    return c2048.bitboard_spawn(ctypes.c_uint64(bitboard))

def bitboard_get_max_tile_func(bitboard):
    return c2048.bitboard_get_max_tile(ctypes.c_uint64(bitboard))

def bitboard_move_func(bitboard, action):
    out_score = ctypes.c_int(0)
    out_moved = ctypes.c_bool(False)
    new_bitboard = c2048.bitboard_move(ctypes.c_uint64(bitboard),
                                      ctypes.c_int(action),
                                      ctypes.byref(out_score),
                                      ctypes.byref(out_moved))
    return new_bitboard, out_score.value, out_moved.value

def bitboard_is_game_over_func(bitboard):
    return c2048.bitboard_is_game_over(ctypes.c_uint64(bitboard))