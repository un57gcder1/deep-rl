from tictactoe import TicTacToe
import random
from pytorch import nn

FULL = []

class Net(nn.Module):
    def __init__(self, x, h, y):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x, h),
            nn.ReLU(),
            nn.Linear(h, y),
        )

    def forward(self, x):
        return self.net(x)

def model(observations):
    pass

def how(game, player):
    board = game.board
    board.append(player)
    move = 11 #move = model(board)
    possible = game.possible()
    if move in possible:
        return ({board : move}, move)
    else:
        move = random.choice(possible)
        return ({board : move}, move)

def play(game):
    data = []
    # A list containing a dictionary: {observations: action},
    # where observations = [game.board, player]
    # and action = move (int)
    player = game.judge()
    if player == "X":
        other = "O"
    else:
        other = "X"

    while game.end() == False:
        if game.turn() == player:
            it = how(game, player)
            data.append(it[0])
            position = it[1]
            game.move(player, position)
        else:
            game.rand_move(other)
    if game.result() == player: # Should ties be included?
        FULL += data

game = TicTacToe()
play(10)
print(FULL)
