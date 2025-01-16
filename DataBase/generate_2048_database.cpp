#ifdef _WIN32
    #include <conio.h>     // for _kbhit(), _getch()
    #include <windows.h>   // optional if you need Sleep, etc.
#else
    #include <sys/select.h>
    #include <termios.h>
    #include <unistd.h>
#endif

bool userWantsStop() {
#ifdef _WIN32
    // Windows approach: use _kbhit() to see if a key was pressed
    if (_kbhit()) {
        int ch = _getch();  // read key
        // if user typed 's' or 'S'
        if (ch == 's' || ch == 'S') {
            return true;
        }
    }
    return false;

#else
    // Unix-like approach using `select()` on STDIN_FILENO
    // 1) Setup an fd_set
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    // 2) Set a zero-timeout (non-blocking)
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    // 3) select(...) tells us if there's data waiting on stdin
    int retval = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    if (retval > 0) {
        // Data is ready => read 1 char
        char ch;
        if (::read(STDIN_FILENO, &ch, 1) > 0) {
            if (ch == 's' || ch == 'S') {
                return true;
            }
        }
    }
    return false;

#endif
}


/*************************************************************
 * generate_2048_database.cpp
 *
 * Demonstrates a bitboard-based 2048 simulation:
 *  - Uses 64-bit to store a 4x4 board (4 bits per cell).
 *  - Precomputed row-merge tables for faster moves.
 *  - Writes (board, move) pairs to a binary file for each game.
 * 
 * We store tile exponents in [0..15], i.e., 2^exponent.
 * 2^(15) = 32768 is the largest tile we handle by default.
 *************************************************************/

#include <bits/stdc++.h>
using namespace std;




// -------------------------------------------------------------------------
// Global Random
// -------------------------------------------------------------------------
static thread_local mt19937_64 g_rng(random_device{}());

// Random range [0..maxExclusive-1]
inline uint64_t randRange(uint64_t maxExclusive) {
    uniform_int_distribution<uint64_t> dist(0, maxExclusive - 1);
    return dist(g_rng);
}

// -------------------------------------------------------------------------
// We store the game data as a vector of <uint64_t, uint8_t> = (bitboard, move)
// This will be eventually written to a binary file.
// move in [0..3], define them as 0=right,1=left,2=down,3=up (arbitrary).
// -------------------------------------------------------------------------
typedef pair<uint64_t, uint8_t> StepData; 

// -------------------------------------------------------------------------
// Each row is 16 bits: 4 cells * 4 bits/cell = 16 bits.
// We create 2 lookup tables: moveLeftTable[] and moveRightTable[].
// Each entry is 32 bits: (gainedScore << 16) | newRow.
// -------------------------------------------------------------------------
static uint32_t moveLeftTable[1 << 16];
static uint32_t moveRightTable[1 << 16];

// Helper: extract the 4 cell exponents from a 16-bit row
inline void extractRow(uint16_t row, array<uint8_t, 4> &cells) {
    cells[0] = (row >>  0) & 0xF;
    cells[1] = (row >>  4) & 0xF;
    cells[2] = (row >>  8) & 0xF;
    cells[3] = (row >> 12) & 0xF;
}

// Helper: build a 16-bit row from 4 cell exponents
inline uint16_t buildRow(const array<uint8_t, 4> &cells) {
    uint16_t row = 0;
    row |= (cells[0] & 0xF) <<  0;
    row |= (cells[1] & 0xF) <<  4;
    row |= (cells[2] & 0xF) <<  8;
    row |= (cells[3] & 0xF) << 12;
    return row;
}

inline uint32_t packRow(uint16_t newRow, uint16_t gainedScore) {
    return ((uint32_t)gainedScore << 16) | (uint32_t)newRow;
}

inline uint16_t getNewRow(uint32_t val) {
    return (uint16_t)(val & 0xFFFF);
}

inline uint16_t getScore(uint32_t val) {
    return (uint16_t)(val >> 16);
}

// Compute the "move left" transformation for one 16-bit row
uint32_t computeMoveLeft(uint16_t row) {
    array<uint8_t, 4> cells;
    extractRow(row, cells);

    uint16_t gainedScore = 0;

    // Step 1: slide left
    int pos = 0;
    for(int i=0; i<4; i++){
        if(cells[i] != 0){
            cells[pos++] = cells[i];
        }
    }
    while(pos < 4) {
        cells[pos++] = 0;
    }

    // Step 2: merge
    for(int i=0; i<3; i++){
        if(cells[i] != 0 && cells[i] == cells[i+1]){
            cells[i]++;
            gainedScore += (1 << cells[i]);
            cells[i+1] = 0;
        }
    }

    // Step 3: slide again
    pos = 0;
    for(int i=0; i<4; i++){
        if(cells[i] != 0){
            cells[pos++] = cells[i];
        }
    }
    while(pos < 4){
        cells[pos++] = 0;
    }

    uint16_t newRow = buildRow(cells);
    return packRow(newRow, gainedScore);
}

// computeMoveRight by reversing row, using computeMoveLeft, then reversing
uint32_t computeMoveRight(uint16_t row) {
    array<uint8_t, 4> cells;
    extractRow(row, cells);
    reverse(cells.begin(), cells.end());
    uint16_t rev = buildRow(cells);

    uint32_t result = computeMoveLeft(rev);
    uint16_t newRev = getNewRow(result);
    uint16_t rowScore = getScore(result);

    extractRow(newRev, cells);
    reverse(cells.begin(), cells.end());
    uint16_t finalRow = buildRow(cells);

    return packRow(finalRow, rowScore);
}

// Build the move tables for all possible 16-bit rows
void initMoveTables() {
    for(int row=0; row < (1<<16); row++){
        moveLeftTable[row]  = computeMoveLeft((uint16_t)row);
        moveRightTable[row] = computeMoveRight((uint16_t)row);
    }
}

// -------------------------------------------------------------------------
// 64-bit Board Layout (4 bits per cell, 16 cells).
// row r is at bits [16*r..16*r+15] in the 64-bit, each row is 16 bits.
// -------------------------------------------------------------------------
inline uint16_t getRow(uint64_t board, int r) {
    return (board >> (16 * r)) & 0xFFFFULL;
}

inline uint64_t setRow(uint64_t board, int r, uint16_t rowVal) {
    uint64_t mask = ((uint64_t)0xFFFF) << (16 * r);
    board &= ~mask;
    board |= ((uint64_t)rowVal << (16 * r));
    return board;
}

inline uint64_t shiftLeft(uint64_t board, int &scoreGained) {
    for(int r=0; r<4; r++){
        uint16_t row = getRow(board, r);
        uint32_t val = moveLeftTable[row];
        uint16_t newRow = getNewRow(val);
        uint16_t rowScore = getScore(val);
        scoreGained += rowScore;
        board = setRow(board, r, newRow);
    }
    return board;
}

inline uint64_t shiftRight(uint64_t board, int &scoreGained) {
    for(int r=0; r<4; r++){
        uint16_t row = getRow(board, r);
        uint32_t val = moveRightTable[row];
        uint16_t newRow = getNewRow(val);
        uint16_t rowScore = getScore(val);
        scoreGained += rowScore;
        board = setRow(board, r, newRow);
    }
    return board;
}

inline uint64_t transpose(uint64_t board) {
    array<uint8_t,16> cells;
    for(int i=0; i<16; i++){
        cells[i] = (board >> (4*i)) & 0xF;
    }
    // Transpose
    array<uint8_t,16> trans;
    for(int r=0; r<4; r++){
        for(int c=0; c<4; c++){
            trans[c*4 + r] = cells[r*4 + c];
        }
    }
    uint64_t newBoard = 0ULL;
    for(int i=0; i<16; i++){
        newBoard |= ((uint64_t)(trans[i] & 0xF) << (4*i));
    }
    return newBoard;
}

inline uint64_t shiftUp(uint64_t board, int &scoreGained) {
    board = transpose(board);
    board = shiftLeft(board, scoreGained);
    board = transpose(board);
    return board;
}

inline uint64_t shiftDown(uint64_t board, int &scoreGained) {
    board = transpose(board);
    board = shiftRight(board, scoreGained);
    board = transpose(board);
    return board;
}

// -------------------------------------------------------------------------
// Place a new tile (exponent=1 or 2) in a random empty cell
// with probability 0.9 for exponent=1, 0.1 for exponent=2
// -------------------------------------------------------------------------
inline uint64_t placeRandomTile(uint64_t board) {
    vector<int> empties;
    empties.reserve(16);
    for(int i=0; i<16; i++){
        uint8_t val = (board >> (4*i)) & 0xF;
        if(val == 0) {
            empties.push_back(i);
        }
    }
    if(empties.empty()) {
        return board; // no empty cell
    }
    int idx = empties[randRange(empties.size())];
    bool is1 = (randRange(10) < 9); 
    uint8_t exponent = (is1 ? 1 : 2); 
    uint64_t mask = 0xFULL << (4*idx);
    board &= ~mask; 
    board |= ((uint64_t)exponent << (4*idx));
    return board;
}

// -------------------------------------------------------------------------
// Check if board is "game over": no empty cell and no merges possible
// -------------------------------------------------------------------------
inline bool isGameOver(uint64_t board) {
    // check empties
    for(int i=0; i<16; i++){
        uint8_t val = (board >> (4*i)) & 0xF;
        if(val == 0) return false;
    }
    // check merges horizontally
    for(int r=0; r<4; r++){
        for(int c=0; c<3; c++){
            uint8_t v1 = (board >> (4*(r*4 + c))) & 0xF;
            uint8_t v2 = (board >> (4*(r*4 + c+1))) & 0xF;
            if(v1 == v2) return false;
        }
    }
    // check merges vertically
    for(int c=0; c<4; c++){
        for(int r=0; r<3; r++){
            uint8_t v1 = (board >> (4*(r*4 + c))) & 0xF;
            uint8_t v2 = (board >> (4*((r+1)*4 + c))) & 0xF;
            if(v1 == v2) return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// Simulate one entire game, returning a list of (bitboard, move).
// We define moves as: 0=right, 1=left, 2=down, 3=up
// 
// Changes made:
//  - We now pick among 4 moves (0..3) instead of 3 (0..2).
//  - We try up to 4 random moves each step. If none changes the board => game over.
//  - We record a step only if the board changes. (board, move) 
//  - We remove the artificial limit totalMoves > 1000 (which could cut off a valid game).
// -------------------------------------------------------------------------
vector<StepData> simulateOneGame() {
    vector<StepData> steps;
    // init board
    uint64_t board = 0ULL;
    // place 2 tiles
    board = placeRandomTile(board);
    board = placeRandomTile(board);

    static uniform_int_distribution<int> moveDist(0,3);

    while(true){
        // If no move can change the board, we are done
        bool anyValidMove = false;
        
        // Try up to 4 times in this step to find a valid move
        for(int tries = 0; tries < 4; tries++){
            uint8_t move = (uint8_t)moveDist(g_rng);
            uint64_t oldBoard = board;
            int gainedScore = 0;
            
            switch(move){
                case 0: // right
                    board = shiftRight(board, gainedScore);
                    break;
                case 1: // left
                    board = shiftLeft(board, gainedScore);
                    break;
                case 2: // down
                    board = shiftDown(board, gainedScore);
                    break;
                case 3: // up
                    board = shiftUp(board, gainedScore);
                    break;
            }
            
            // If the board changed, place a random tile and record the step
            if(board != oldBoard) {
                // Record oldBoard & the move that changed it
                steps.push_back({oldBoard, move});

                board = placeRandomTile(board);
                
                anyValidMove = true;
                break; // exit the tries-loop, go to next step
            } else {
                // revert board to old if it didn't change
                board = oldBoard;
            }
        }

        // If none of the 4 random tries changed the board, we're stuck => game over
        if(!anyValidMove) {
            break;
        }

        // If we are truly out of moves (game over), break
        if(isGameOver(board)){
            break;
        }
    }

    return steps;
}

// -------------------------------------------------------------------------
// We'll write a function to store many games in a single file
// [uint64_t numGames]
// Then for each game:
//   [uint32_t steps]
//   For each step: [uint64_t bitboard], [uint8_t move]
// -------------------------------------------------------------------------
void writeAllGamesBinary(const string &filename,
                         const vector<vector<StepData>> &allGames)
{
    ofstream out(filename, ios::binary);
    if(!out){
        cerr << "Cannot open file for writing: " << filename << endl;
        return;
    }

    // 1) Write numGames (64-bit)
    uint64_t numGames = allGames.size();
    out.write(reinterpret_cast<const char*>(&numGames), sizeof(numGames));

    // 2) For each game
    for(const auto &game : allGames) {
        // #steps = (uint32_t)game.size()
        uint32_t steps = (uint32_t)game.size();
        out.write(reinterpret_cast<const char*>(&steps), sizeof(steps));
        for(const auto &step : game){
            // step.first = bitboard, step.second=move
            uint64_t board = step.first;
            uint8_t  move  = step.second;
            out.write(reinterpret_cast<const char*>(&board), sizeof(board));
            out.write(reinterpret_cast<const char*>(&move),  sizeof(move));
        }
    }
    out.close();
}

bool hasMinimumHighTile(const vector<StepData>& gameData) {
    if (gameData.empty()) return false;
    
    // Get the final board state (the very last move won't have
    // a "resulting" board stored, so let's see the last board in the steps).
    // The last stored board is stepData.back().first
    uint64_t finalBoard = gameData.back().first;
    
    // Find highest tile by checking each 4-bit segment
    int highestTilePower = 0;
    for (int i = 0; i < 16; i++) {
        int tilePower = (finalBoard >> (i * 4)) & 0xF;
        highestTilePower = max(highestTilePower, tilePower);
    }
    
    // Check if highest tile power is at least 9 (512 = 2^9)
    return highestTilePower >= 9;
}




// -------------------------------------------------------------------------
// Worker function: each thread simulates up to `gamesNeeded` valid games 
// or stops early if `stopRequested` is set. It returns a local vector of 
// valid games from this thread. 
// 
// NOTE: The function updates a shared 'storedCount' atomic whenever a game 
// is successfully stored.
// -------------------------------------------------------------------------
static vector<vector<pair<uint64_t, uint8_t>>>
workerFunc(int gamesNeeded, atomic_bool &stopRequested,
           atomic<int> &storedCount)
{
    vector<vector<pair<uint64_t, uint8_t>>> localResults;
    localResults.reserve(gamesNeeded);

    int localStored = 0;
    while (localStored < gamesNeeded && !stopRequested.load()) {
        // Simulate one game
        auto gameData = simulateOneGame();

        // Check if it meets the threshold
        if (hasMinimumHighTile(gameData)) {
            localResults.push_back(std::move(gameData));
            localStored++;
            // Increment the global count of stored games
            storedCount.fetch_add(1, memory_order_relaxed);
        }
    }
    return localResults;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

#ifdef __unix__
    enableRawMode();  // optional
#endif

    // 1) Initialize row-move tables
    initMoveTables();

    // 2) Desired total number of valid games, and number of threads
    const int NUM_GAMES   = 10000000;  // total desired
    const int NUM_THREADS = max(1u, thread::hardware_concurrency());
    cout << "Using " << NUM_THREADS << " threads.\n";

    // We'll split the total “valid” games across threads evenly
    int gamesPerThread = NUM_GAMES / NUM_THREADS;

    // 3) Create shared flags/counters
    atomic_bool stopRequested(false);
    // Remove 'static' to allow capturing in lambdas
    atomic<int> storedCount(0);

    // 4) Input-checking thread: periodically checks userWantsStop()
    thread inputThread([&stopRequested]() {
        while (!stopRequested.load()) {
            if (userWantsStop()) {
                stopRequested.store(true);
                break;
            }
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    });

    // 5) Progress-print thread: prints whenever we cross multiples of 100 
    // stored games, or if the user stops the run.
    thread progressThread([&stopRequested, &storedCount, NUM_GAMES]() {
        // We'll keep track of the time to measure how long we've been generating.
        auto startTime = chrono::steady_clock::now();
        int lastPrintedCount = 0;

        while (!stopRequested.load()) {
            // Current total # of stored games
            int sc = storedCount.load(memory_order_relaxed);

            // Print if we've reached 100 more than last time
            if (sc - lastPrintedCount >= 100) {
                auto now = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - startTime).count();
                cout << "\rStored " << sc << " / " << NUM_GAMES 
                     << " games in " << elapsed << "s";
                cout.flush();
                lastPrintedCount = sc;
            }

            // If we've already reached the goal, or if user requested a stop, break
            if (sc >= NUM_GAMES) {
                break;
            }
            if (stopRequested.load()) {
                break;
            }
            // Sleep a bit to avoid busy-wait
            this_thread::sleep_for(chrono::milliseconds(200));
        }

        // Final print to ensure all games are reported
        int finalCount = storedCount.load(memory_order_relaxed);
        auto now = chrono::steady_clock::now();
        auto totalElapsed = chrono::duration_cast<chrono::seconds>(now - startTime).count();
        cout << "\rStored " << finalCount << " / " << NUM_GAMES 
             << " games in " << totalElapsed << "s\n";
    });

    // 6) Launch worker threads (async)
    vector<future<vector<vector<pair<uint64_t, uint8_t>>>>> futures;
    futures.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        futures.push_back(
            std::async(std::launch::async, [=, &stopRequested, &storedCount]() {
                return workerFunc(gamesPerThread, stopRequested, storedCount);
            })
        );
    }

    // 7) Collect results from all threads
    vector<vector<pair<uint64_t, uint8_t>>> allGames;
    allGames.reserve(NUM_GAMES);

    for (auto &f : futures) {
        auto partial = f.get(); // blocks until that thread finishes
        for (auto &g : partial) {
            allGames.push_back(std::move(g));
        }
    }

    // Once workers are done or user requested stop, we can join the 
    // progress- and input-threads so they exit gracefully.
    if (progressThread.joinable()) {
        progressThread.join();
    }
    if (inputThread.joinable()) {
        inputThread.join();
    }

    // 8) Write results to a binary file
    const string filename = "games.bin";
    writeAllGamesBinary(filename, allGames);

#ifdef __unix__
    disableRawMode();  // optional
#endif

    cout << "\n\nStopped run.\n"
         << "Wrote " << allGames.size() 
         << " games to '" << filename << "'\n";

    return 0;
}