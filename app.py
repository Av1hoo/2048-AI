import streamlit as st
import matplotlib.pyplot as plt  
from streamlit_shortcuts import add_keyboard_shortcuts

from CytonFiles.Game_Logic import Game2048
from CytonFiles.Batch_Games import Batch_Games
from CytonFiles.Visualizer import Visualizer

# -------------- CONFIG --------------
AVAILABLE_STRATEGIES = ["Expectimax", "Minimax", "Random", "DQN"]
DEFAULT_DEPTHS = {
    "Expectimax": 2,
    "Minimax": 2
}

# -------------- HELPER FUNCTIONS --------------
def init_game_state():
    """
    Initializes a fresh Game2048 object in session state if it doesn't exist.
    """
    if "game" not in st.session_state:
        # Default to Expectimax with best_game.pkl as in your logic
        st.session_state["game"] = Game2048(ai_strategy="Expectimax")
        st.session_state["depth"] = DEFAULT_DEPTHS["Expectimax"]
        st.session_state["game"].reset_game()
        st.session_state["ai_playing"] = False  # Flag to indicate AI is playing
        st.session_state["batch_results"] = None  # Initialize batch results
    

def update_ai_strategy():
    """
    Re-initialize the Game2048 with the selected strategy.
    """
    strategy = st.session_state["chosen_strategy"]
    # If the strategy is Expectimax or Minimax, set a default depth
    if strategy in DEFAULT_DEPTHS:
        st.session_state["depth"] = DEFAULT_DEPTHS[strategy]
    else:
        st.session_state["depth"] = 0

    # Create a new game object with the selected strategy
    st.session_state["game"] = Game2048(ai_strategy=strategy)
    # Optionally, set depth if your Game2048 class supports it
    game = st.session_state["game"]
    if hasattr(game, 'set_depth'):
        game.set_depth(st.session_state["depth"])
    game.reset_game()

def generate_board_html(game: Game2048) -> str:
    """
    Generates HTML for the 4x4 game board.
    """
    board_data = game.get_board_2d()

    # Define cell colors consistent with your original GUI
    cell_colors = {
        0: "#CDC1B4",
        2: "#EEE4DA",
        4: "#EDE0C8",
        8: "#F2B179",
        16: "#F59563",
        32: "#F67C5F",
        64: "#F65E3B",
        128: "#EDCF72",
        256: "#EDCC61",
        512: "#EDC850",
        1024: "#EDC53F",
        2048: "#EDC22E",
        4096: "#6BC910",
        8192: "#63BE07"
    }

    def color_cell(val):
        color = cell_colors.get(val, "#3C3A32")  # Default color for unknown values
        text_color = "black" if val in [2, 4] else "white"  # Improve readability
        return f"background-color: {color}; color: {text_color}; font-weight: bold;"

    cell_html = ''.join([
        f"""
        <div style="
            width: 100px;
            height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            border-radius: 5px;
            {color_cell(val)}
        ">
            {val if val > 0 else ''}
        </div>
        """ for row in board_data for val in row
    ])

    return f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, 100px);
        grid-template-rows: repeat(4, 100px);
        gap: 10px;
        justify-content: center;
        align-items: center;
        margin: auto;
    ">
        {cell_html}
    </div>
    """

def show_board(game: Game2048, board_placeholder):
    """
    Displays the game board using Streamlit's markdown with HTML.
    """
    board_html = generate_board_html(game)
    board_placeholder.html(board_html)

def ai_step():
    """
    Perform a single AI step.
    """
    game = st.session_state["game"]
    moved = game.ai_single_step()
    if game.is_game_over():
        st.session_state["ai_playing"] = False  # Stop AI if no move or game over

def move_up():
    st.session_state["game"].do_move(0)
    st.session_state["ai_playing"] = False  # Stop AI if user makes a move

def move_down():
    st.session_state["game"].do_move(1)
    st.session_state["ai_playing"] = False

def move_left():
    st.session_state["game"].do_move(2)
    st.session_state["ai_playing"] = False

def move_right():
    st.session_state["game"].do_move(3)
    st.session_state["ai_playing"] = False

# -------------- STREAMLIT APP --------------
def main():
    st.set_page_config(layout="wide", page_title="2048 Streamlit Demo")

    # Initialize game state
    init_game_state()

    add_keyboard_shortcuts({
        "ArrowUp": move_up,
        "ArrowDown": move_down,
        "ArrowLeft": move_left,
        "ArrowRight": move_right,
        "w": move_up,
        "s": move_down,
        "a": move_left,
        "d": move_right,
    })
    # Create 3 tabs (Streamlit >= 1.10.0 has st.tabs)
    tabs = st.tabs(["Play 2048", "Run Batch", "Batch Results"])

    # ------------- TAB 1: Play 2048 -------------
    with tabs[0]:
        st.header("Play 2048")

        game: Game2048 = st.session_state["game"]

        # Let user pick AI Strategy
        strategy = st.selectbox(
            "AI Strategy:",
            AVAILABLE_STRATEGIES,
            index=AVAILABLE_STRATEGIES.index(game.ai_strategy) if game.ai_strategy in AVAILABLE_STRATEGIES else 0,
            key="chosen_strategy",
            on_change=update_ai_strategy
        )

        # If strategy is Minimax or Expectimax, pick depth
        if strategy in ["Minimax", "Expectimax"]:
            st.session_state["depth"] = st.number_input(
                "Search Depth:",
                min_value=1,
                max_value=10,
                value=st.session_state["depth"],
                key="depth_input"
            )
            # Assuming Game2048 can accept depth, else handle accordingly
            if hasattr(game, 'set_depth'):
                game.set_depth(st.session_state["depth"])
        else:
            st.session_state["depth"] = 0  # Reset depth for irrelevant strategies
            if hasattr(game, 'set_depth'):
                game.set_depth(0)  # Assuming set_depth method exists

        
        # Buttons for user to control
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            if st.button("↑ Up", "ArrowUp"):
                game.do_move(0)
        with col2:
            if st.button("↓ Down"):
                game.do_move(1)
        with col3:
            if st.button("← Left"):
                game.do_move(2)
        with col4:
            if st.button("→ Right"):
                game.do_move(3)


        # AI step & AI full
        st.session_state["max_ai_moves"] = 5000  # Limit AI to 100 moves
        c1, c2 = st.columns(2)
        board_placeholder = st.empty()
        with c1:
            if st.button("AI Step"):
                ai_step()
        with c2:
            if st.button("AI Game"):
                st.session_state["ai_playing"] = True
                for _ in range(st.session_state["max_ai_moves"]):
                    if not st.session_state["ai_playing"]:
                        break
                    ai_step()
                    board_html = generate_board_html(game)
                    # put it in the cecnter
                    board_placeholder.html(board_html)

        # Restart
        if st.button("Restart Game"):
            st.session_state["game"].reset_game()
            st.session_state["ai_playing"] = False

        # If AI is playing, perform a step
        if st.session_state.get("ai_playing"):
            ai_step()

        # Show the board
        show_board(game, board_placeholder)
        # Display scores
        st.write(f"**Score**: {game.get_score()} | **Highest Tile**: {game.get_highest()}")
        if game.is_game_over():
            st.error("Game Over!")

    # ------------- TAB 2: Run Batch -------------
    with tabs[1]:
        st.header("Run Batch Experiments")

        # Options for single or multiple batch
        mode = st.radio("Choose Batch Mode", ["Single Strategy", "Multiple Strategies"])
        num_games = st.number_input("Number of Games:", min_value=1, max_value=1000, value=10)

        # Initialize STRATEGY_BATCH in session state if not present
        if "STRATEGY_BATCH" not in st.session_state:
            st.session_state["STRATEGY_BATCH"] = {
                "Expectimax": 2,
                "Minimax": 2,
                "Random": 0,
                "DQN": 0,
            }

        if mode == "Single Strategy":
            # Single strategy: pick from the list
            single_strat = st.selectbox("Which Strategy?", AVAILABLE_STRATEGIES, key="single_strategy_select")

            depth_val = 0
            if single_strat in DEFAULT_DEPTHS:
                depth_val = st.number_input(
                    "Depth for Single Strategy",
                    min_value=1,
                    max_value=10,
                    value=DEFAULT_DEPTHS[single_strat],
                    key="single_strategy_depth"
                )

            if st.button("Run Single Batch"):
                # Call your batch function
                with st.spinner("Running batch..."):
                    stats = Batch_Games.run_batch_games(
                        num_games=num_games,
                        strategy=single_strat,
                        MODEL_PATH="model.pth", 
                        DEPTH=depth_val
                    )
                st.success("Batch completed!")
                # Store in session_state for tab 3
                st.session_state["batch_results"] = stats
                st.session_state["num_games"] = num_games  # Store num_games

        else:
            # Multiple strategies
            # Let user pick which strategies to run
            chosen_strategies = st.multiselect(
                "Select Strategies",
                AVAILABLE_STRATEGIES,
                default=["Expectimax","Minimax"],
                key="multiple_strategy_select"
            )
            # For each chosen strategy, let them pick a depth if relevant
            multi_depths = {}
            for strat in chosen_strategies:
                if strat in DEFAULT_DEPTHS:
                    d = st.number_input(
                        f"Depth for {strat}",
                        min_value=1,
                        max_value=10,
                        value=DEFAULT_DEPTHS[strat],
                        key=f"depth_{strat}"
                    )
                    multi_depths[strat] = d
                else:
                    multi_depths[strat] = 0

            if st.button("Run Multiple Batch"):
                # Build a dictionary for STRATEGY_BATCH like: {"Expectimax": 2, "Minimax": 4, "Random": 0, ...}
                strategy_batch = {s: multi_depths[s] for s in chosen_strategies}

                with st.spinner("Running multiple batch..."):
                    stats_all = Batch_Games.run_multiple_batch_strategies(num_games, strategy_batch, "model.pth")

                st.success("All Batches completed!")
                # Store in session state
                st.session_state["batch_results"] = stats_all
                st.session_state["num_games"] = num_games  # Store num_games

    # ------------- TAB 3: Batch Results -------------
    with tabs[2]:
        st.header("Batch Results / Visualization")

        if "batch_results" not in st.session_state or st.session_state["batch_results"] is None:
            st.warning("No batch results to show. Please run a batch first in the second tab.")
        else:
            results_data = st.session_state["batch_results"]
            num_games = st.session_state.get("num_games", "N/A")  # Retrieve num_games

            st.subheader("Visualization")
            with st.spinner("Generating charts..."):
                try:
                    figures = Visualizer.visualize_stats(results_data, num_games)
                    # Ensure figures is a list
                    if isinstance(figures, list):
                        for fig in figures:
                            # limit size of the figure
                            fig.set_size_inches(4, 3)
                            st.pyplot(fig, use_container_width=False)
                    elif isinstance(figures, plt.Figure):
                        st.pyplot(figures)
                    else:
                        st.error("The generated figure is not a valid Matplotlib figure.")
                    st.success("Charts generated!")
                except Exception as e:
                    st.error(f"An error occurred while generating the chart: {e}")

            # show it folded
            st.subheader("Raw Results")
            st.json(results_data, expanded=False)

# -------------- ENTRY POINT --------------
if __name__ == "__main__":
    main()