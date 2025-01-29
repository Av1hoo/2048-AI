# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from CytonFiles.Game_Logic import Game2048
import time
import os
from CytonFiles.Visualizer import Visualizer
from CytonFiles.Batch_Games import Batch_Games

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")  # Use environment variable

AVAILABLE_STRATEGIES = ["Expectimax", "Minimax", "Random", "DQN"]

# Initialize a global variable to store batch results
# In production, consider using a database or persistent storage
batch_results = {
    "single": None,      # Single batch run
    "multiple": []       # Multiple batch runs
}

def load_game_from_session():
    if "game_state" in session:
        state = session["game_state"]
        required_keys = ['bitboard', 'score', 'highest', 'ai_strategy', 'depth']
        if all(key in state for key in required_keys):
            game = Game2048(
                ai_strategy=state.get("ai_strategy", "Expectimax"),
                depth=state.get("depth", 4)
            )
            game.set_state(state)
            return game
        else:
            # Handle incomplete state, e.g., reset the game
            return None
    return None




def save_game_to_session(game):
    """Save the Game2048 instance to the session."""
    session["game_state"] = game.get_state()

@app.route("/")
def home():
    """Redirect to the main game page."""
    return redirect(url_for("game"))

@app.route("/game", methods=["GET"])
def game():
    # Load from session, if it exists
    game = load_game_from_session()

    # Grab any GET parameters
    requested_strategy = request.args.get('strategy')
    requested_depth = request.args.get('depth', type=int)

    # If no game in session, create a new one (use default or requested params)
    if not game:
        strategy = request.args.get('strategy', "Expectimax")  # Defaults or GET params
        depth = request.args.get('depth', 4, type=int)
        game = Game2048(ai_strategy=strategy, depth=depth)
        game.reset_game()
        save_game_to_session(game)
    else:
        # If the user specifically passed new strategy/depth, re-init the game
        if requested_strategy and requested_strategy != game.ai_strategy:
            game = Game2048(ai_strategy=requested_strategy, depth=requested_depth)
            if requested_strategy in ["Expectimax", "Minimax"] and requested_depth:
                game.set_depth(requested_depth)
            game.reset_game()
            save_game_to_session(game)
        elif requested_depth is not None:
            # If same strategy but different depth, re-init accordingly
            current_depth = game.factory.EXPECTIMAX_DEPTH if game.ai_strategy == "Expectimax" else \
                            game.factory.MINIMAX_DEPTH if game.ai_strategy == "Minimax" else 0
            if requested_depth != current_depth:
                strategy = game.ai_strategy
                game = Game2048(ai_strategy=strategy, depth=requested_depth)
                if strategy in ["Expectimax", "Minimax"]:
                    game.set_depth(requested_depth)
                game.reset_game()
                save_game_to_session(game)

    # Now just read from the current game in session
    board = game.get_board_2d()
    score = game.get_score()
    highest = game.get_highest()
    game_over = game.is_game_over()
    current_ai = game.ai_strategy
    # For displaying on the page, get the "current" depth from your game
    if current_ai == "Expectimax":
        depth = game.factory.EXPECTIMAX_DEPTH
    elif current_ai == "Minimax":
        depth = game.factory.MINIMAX_DEPTH
    else:
        depth = 0

    return render_template(
        "game.html",
        board=board,
        score=score,
        highest=highest,
        game_over=game_over,
        strategies=AVAILABLE_STRATEGIES,
        current_ai=current_ai,
        depth=depth
    )


@app.route("/move/<direction>", methods=["POST"])
def move(direction):
    """
    Handle user moves: up, down, left, right.
    """
    game = load_game_from_session()
    if game is None:
        return jsonify({"error": "No game in session. Please reload the game."}), 400

    move_map = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3
    }

    if direction in move_map:
        game.do_move(move_map[direction])

    save_game_to_session(game)
    return jsonify({
        "status": "ok",
        "score": game.get_score(),
        "highest": game.get_highest(),
        "game_over": game.is_game_over(),
        "board": game.get_board_2d()
    })

@app.route("/reset", methods=["POST"])
def reset_game():
    """
    Reset the game entirely.
    """
    game = load_game_from_session()
    if game is None:
        return jsonify({"error": "No game in session."}), 400

    # Reset the game with the current AI settings
    ai_strategy = game.ai_strategy
    depth = game.factory.EXPECTIMAX_DEPTH if ai_strategy == "Expectimax" else game.factory.MINIMAX_DEPTH if ai_strategy == "Minimax" else 0
    game = Game2048(ai_strategy=ai_strategy, depth=depth)
    if ai_strategy in ["Expectimax", "Minimax"]:
        game.set_depth(depth)
    game.reset_game()
    save_game_to_session(game)

    return jsonify({
        "status": "reset",
        "score": game.get_score(),
        "highest": game.get_highest(),
        "game_over": game.is_game_over(),
        "board": game.get_board_2d()
    })

@app.route("/ai_step", methods=["POST"])
def ai_step():
    """
    Perform a single AI move.
    """
    game = load_game_from_session()
    if game is None:
        return jsonify({"error": "No game in session."}), 400

    game.ai_single_step()
    save_game_to_session(game)

    return jsonify({
        "status": "done",
        "score": game.get_score(),
        "highest": game.get_highest(),
        "game_over": game.is_game_over(),
        "board": game.get_board_2d()
    })

@app.route("/play_full_ai", methods=["POST"])
def play_full_ai():
    """
    Initiate the AI to play the game to completion with step-by-step updates.
    Handled entirely on the client-side.
    """
    return jsonify({
        "status": "ready_to_play_full_ai"
    })

@app.route("/settings", methods=["POST"])
def update_settings():
    """
    Update AI strategy and depth settings.
    Resets the game upon changing settings.
    """
    strategy = request.form.get("strategy", "Expectimax")
    depth = int(request.form.get("depth", 4))
    
    # Create a new game instance with updated settings
    game = Game2048(ai_strategy=strategy, depth=depth)
    if strategy in ["Expectimax", "Minimax"]:
        game.set_depth(depth)
    game.reset_game()
    
    # Save the complete game state to the session
    save_game_to_session(game)
    
    return redirect(url_for("game"))


@app.route("/batch", methods=["GET"])
def batch():
    """
    Display the batch run configuration page.
    """
    return render_template("batch.html", strategies=AVAILABLE_STRATEGIES)

@app.route("/run_batch", methods=["POST"])
def run_batch():
    """
    Run a single batch of games based on user configuration.
    """
    strategy = request.form.get("strategy")
    depth_str = request.form.get("depth", "2")
    num_games_str = request.form.get("num_games", "1000")

    try:
        depth = int(depth_str)
        num_games = int(num_games_str)
    except ValueError:
        depth = 2
        num_games = 1000

    start_time = time.time()
    stats = Batch_Games.run_batch_games(
        num_games=num_games,
        strategy=strategy,
        MODEL_PATH="model.pth",
        DEPTH=depth
    )
    end_time = time.time()
    time_taken = end_time - start_time

    # Visualize stats and save plots
    plot_paths = Visualizer.visualize_stats(stats, num_games, show_plots=False)

    # Store batch run results
    batch_results["single"] = {
        "strategy": strategy,
        "depth": depth,
        "num_games": num_games,
        "time_taken": f"{time_taken:.2f}",
        "avg_score_plot": plot_paths.get('avg_score_plot'),
        "best_score_plot": plot_paths.get('best_score_plot'),
        "tile_distribution_plot": plot_paths.get('tile_distribution_plot')
    }

    return redirect(url_for("results"))

@app.route("/run_multiple_batch", methods=["POST"])
def run_multiple_batch():
    """
    Run multiple batches of games based on user configuration.
    """
    strategies_str = request.form.get("strategies", "")
    depths_str = request.form.get("depths", "")
    num_games_str = request.form.get("num_games", "1000")

    strategies = [s.strip() for s in strategies_str.split(",") if s.strip() in AVAILABLE_STRATEGIES]
    depths = [int(d.strip()) if d.strip().isdigit() else 0 for d in depths_str.split(",")]
    num_games = int(num_games_str) if num_games_str.isdigit() else 1000

    # Ensure that strategies and depths lists are of the same length
    if len(depths) < len(strategies):
        depths += [0] * (len(strategies) - len(depths))
    else:
        depths = depths[:len(strategies)]

    strategy_depth_map = dict(zip(strategies, depths))

    start_time = time.time()
    stats_all = Batch_Games.run_multiple_batch_strategies(
        num_games=num_games,
        strategies=strategy_depth_map,
        MODEL_PATH="model.pth"
    )
    end_time = time.time()
    time_taken = end_time - start_time

    # Visualize stats and save plots
    plot_paths = Visualizer.visualize_stats(stats_all, num_games, show_plots=False)

    # Prepare multiple batch results
    multiple_batches = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for strategy, depth in strategy_depth_map.items():
        multiple_batches.append({
            "strategy": strategy,
            "depth": depth,
            "num_games": num_games,
            "time_taken": f"{time_taken:.2f}",
            "avg_score_plot": plot_paths.get('avg_score_plot'),
            "best_score_plot": plot_paths.get('best_score_plot'),
            "tile_distribution_plot": plot_paths.get('tile_distribution_plot')
        })

    batch_results["multiple"] = multiple_batches

    return redirect(url_for("results"))

@app.route("/results", methods=["GET"])
def results():
    """
    Display the results of batch runs.
    """
    single_batch = batch_results.get("single")
    multiple_batches = batch_results.get("multiple")
    return render_template("results.html", 
                           single_batch=single_batch, 
                           multiple_batches=multiple_batches)

if __name__ == "__main__":
    # Ensure the 'static/Stats' directory exists
    if not os.path.exists("static/Stats"):
        os.makedirs("static/Stats")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
