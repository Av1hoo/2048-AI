// static/script.js

document.addEventListener("DOMContentLoaded", () => {
  let isPlayingFullAI = false;
  let aiPlayTimeout = null;

  // Listen to arrow keys for user moves
  document.addEventListener("keydown", (e) => {
    if (isPlayingFullAI) return; // Prevent user moves during AI run
    if (e.key === "ArrowUp") {
      sendMove("up");
    } else if (e.key === "ArrowDown") {
      sendMove("down");
    } else if (e.key === "ArrowLeft") {
      sendMove("left");
    } else if (e.key === "ArrowRight") {
      sendMove("right");
    }
  });

  // Reset game
  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      fetch("/reset", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            alert(data.error);
            return;
          }
          isPlayingFullAI = false;
          updateUI(data);
        });
    });
  }

  // Single AI move
  const aiStepBtn = document.getElementById("aiStepBtn");
  if (aiStepBtn) {
    aiStepBtn.addEventListener("click", () => {
      fetch("/ai_step", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            alert(data.error);
            return;
          }
          updateUI(data);
        });
    });
  }

  // Play Full AI
  const playAiBtn = document.getElementById("playAiBtn");
  const stopAiBtn = document.getElementById("stopAiBtn");
  if (playAiBtn && stopAiBtn) {
    playAiBtn.addEventListener("click", () => {
      if (isPlayingFullAI) return;
      isPlayingFullAI = true;
      playAiBtn.style.display = "none";
      stopAiBtn.style.display = "inline-block";
      playFullAI(); // Start the AI loop
    });

    stopAiBtn.addEventListener("click", () => {
      isPlayingFullAI = false;
      playAiBtn.style.display = "inline-block";
      stopAiBtn.style.display = "none";
      if (aiPlayTimeout) {
        clearTimeout(aiPlayTimeout);
        aiPlayTimeout = null;
      }
    });
  }

  // Function to send user move
  function sendMove(direction) {
    fetch(`/move/${direction}`, {
      method: "POST"
    })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      updateUI(data);
    });
  }

  // Function to update the UI based on server response
  function updateUI(data) {
    // Update score and highest
    document.getElementById("score").textContent = data.score;
    document.getElementById("highest").textContent = data.highest;

    // Update game over message
    const gameOverElement = document.querySelector(".game-over");
    if (data.game_over) {
      if (!gameOverElement) {
        const newGameOver = document.createElement("p");
        newGameOver.className = "game-over";
        newGameOver.textContent = "Game Over!";
        document.querySelector(".info").appendChild(newGameOver);
      }
    } else {
      if (gameOverElement) {
        gameOverElement.remove();
      }
    }

    // Update the board
    const boardDiv = document.querySelector(".board");
    const newBoardHTML = generateBoardHTML(data.board);
    boardDiv.innerHTML = newBoardHTML;

    // Optionally, you can add animations or transitions here
  }

  // Function to generate HTML for the board based on the board state
  function generateBoardHTML(board) {
    const tileColors = {
      0:"#CDC1B4", 2:"#EEE4DA", 4:"#EDE0C8", 8:"#F2B179",
      16:"#F59563", 32:"#F67C5F", 64:"#F65E3B", 128:"#EDCF72",
      256:"#EDCC61", 512:"#EDC850", 1024:"#EDC53F", 2048:"#EDC22E",
      4096:"#6BC910", 8192:"#63BE07"
    };

    let html = '';
    board.forEach(row => {
      html += '<div class="row">';
      row.forEach(cell => {
        const color = tileColors[cell] || "#CDC1B4";
        html += `<div class="cell" style="background-color: ${color};">
                  ${cell > 0 ? cell : ''}
                </div>`;
      });
      html += '</div>';
    });
    return html;
  }

  // Function to perform Play Full AI with 30ms delay between moves
  function playFullAI() {
    if (!isPlayingFullAI) return; // Double-check the flag before proceeding

    fetch("/ai_step", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          alert(data.error);
          isPlayingFullAI = false;
          toggleAIButtons();
          return;
        }
        updateUI(data);
        if (!data.game_over && isPlayingFullAI) {
          aiPlayTimeout = setTimeout(playFullAI, 30); // 30ms delay
        } else {
          // Re-enable the Play Full AI button after game over or stop
          isPlayingFullAI = false;
          toggleAIButtons();
        }
      })
      .catch(error => {
        console.error("Error during AI move:", error);
        isPlayingFullAI = false;
        toggleAIButtons();
      });
  }

  // Helper function to toggle Play and Stop AI buttons
  function toggleAIButtons() {
    const playAiBtn = document.getElementById("playAiBtn");
    const stopAiBtn = document.getElementById("stopAiBtn");
    if (playAiBtn && stopAiBtn) {
      playAiBtn.style.display = isPlayingFullAI ? "none" : "inline-block";
      stopAiBtn.style.display = isPlayingFullAI ? "inline-block" : "none";
    }
  }
});
