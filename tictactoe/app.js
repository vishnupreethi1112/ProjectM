document.addEventListener("DOMContentLoaded", () => {
  let boxes = document.querySelectorAll(".box");
  let resetBtn = document.querySelector("#reset-btn");
  let newGameBtn = document.querySelector("#new-btn");
  let msgContainer = document.querySelector(".msg-container");
  let msg = document.querySelector("#msg");

  let turnO = true; // playerO starts
  let count = 0; // To Track Draw

  const winPatterns = [
    [0, 1, 2],
    [0, 3, 6],
    [0, 4, 8],
    [1, 4, 7],
    [2, 5, 8],
    [2, 4, 6],
    [3, 4, 5],
    [6, 7, 8],
  ];

  const resetGame = () => {
    turnO = true;
    count = 0;
    enableBoxes();
    msgContainer.classList.add("hide");
  };

  boxes.forEach((box) => {
    box.addEventListener("click", () => {
      if (box.disabled) return;

      if (turnO) {
        // playerO
        box.innerText = "O";
        turnO = false;
      } else {
        // playerX
        box.innerText = "X";
        turnO = true;
      }
      box.disabled = true;
      count++;

      let isWinner = checkWinner();

      if (isWinner) {
        let winnerName = prompt(`Congratulations ${box.innerText}! You won! Please enter your name:`);
        if (winnerName) {
          msg.innerText = `Congratulations, ${winnerName}! Winner is ${box.innerText}`;
        } else {
          msg.innerText = `Congratulations, Winner is ${box.innerText}`;
        }
        msgContainer.classList.remove("hide");
        disableBoxes();
      } else if (count === 9) {
        gameDraw();
      }
    });
  });

  const gameDraw = () => {
    msg.innerText = `Game was a Draw.`;
    msgContainer.classList.remove("hide");
    disableBoxes();
  };

  const disableBoxes = () => {
    for (let box of boxes) {
      box.disabled = true;
    }
  };

  const enableBoxes = () => {
    for (let box of boxes) {
      box.disabled = false;
      box.innerText = "";
    }
  };

  const checkWinner = () => {
    for (let pattern of winPatterns) {
      let pos1Val = boxes[pattern[0]].innerText;
      let pos2Val = boxes[pattern[1]].innerText;
      let pos3Val = boxes[pattern[2]].innerText;

      if (pos1Val !== "" && pos2Val !== "" && pos3Val !== "") {
        if (pos1Val === pos2Val && pos2Val === pos3Val) {
          return true;
        }
      }
    }
    return false;
  };

  newGameBtn.addEventListener("click", resetGame);
  resetBtn.addEventListener("click", resetGame);
});
