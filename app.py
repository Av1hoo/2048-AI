# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from CytonFiles.Game_Logic import Game2048

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_SOME_RANDOM_SECRET_KEY"  # Needed for session encryption

# Use a single global object for the AI strategy, or re-create each time. 
# For demonstration, we won't store large objects in the session directly. 

@app.route("/")
def index():
    """
    Display the current game board. If there's no game in the session, create one.
    """
    if "game_state" not in session:
        # Create a new game
        game = Game2048(ai_strategy="Expectimax")  # or whichever strategy you prefer
        game.reset_game()
        session["game_state"] = game.get_state()
    else:
        # Load existing game
        game = Game2048(ai_strategy="Expectimax")
        game.set_state(session["game_state"])

    board = game.get_board_2d()
    score = game.get_score()
    highest = game.get_highest()
    game_over = game.is_game_over()

    return render_template("index.html", 
                           board=board, 
                           score=score, 
                           highest=highest, 
                           game_over=game_over)

@app.route("/move/<direction>")
def move(direction):
    """
    Handle user moves: up, down, left, right.
    direction can be: 'up', 'down', 'left', or 'right'
    """
    if "game_state" not in session:
        return redirect(url_for("index"))

    # Restore the game
    game = Game2048(ai_strategy="Expectimax")
    game.set_state(session["game_state"])

    # Map direction to internal moves
    move_map = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3
    }
    if direction in move_map:
        game.do_move(move_map[direction])
        session["game_state"] = game.get_state()

    return redirect(url_for("index"))

@app.route("/reset")
def reset_game():
    """
    Reset the game entirely.
    """
    game = Game2048(ai_strategy="Expectimax")
    game.reset_game()
    session["game_state"] = game.get_state()
    return redirect(url_for("index"))

@app.route("/ai-move")
def ai_move():
    """
    Let the AI play one step.
    """
    if "game_state" not in session:
        return redirect(url_for("index"))
    game = Game2048(ai_strategy="Expectimax")
    game.set_state(session["game_state"])
    game.ai_single_step()
    session["game_state"] = game.get_state()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
