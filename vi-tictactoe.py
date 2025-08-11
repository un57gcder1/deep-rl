from tictactoe import TicTacToe

# implement value iteration method

rt = {} # key: init-state+action+dest-state, val: immediate reward
tt = {} # key: init-state+action, val: {key: dest-state, val: count}
vt = {} # key: state, val: value

def play(game):
    player = game.judge()
    if player == "X":
        other = "O"
    else:
        other = "X"

    while game.end() == False:
        state = game.board[:]
        state.append(player)
        state = tuple(state)
        if game.turn() == player:
            possible_actions = game.possible()
            for p in possible_actions:
                state_plus = state + (p,)
                if state_plus in tt.keys():
                    pass # do more edits
                else:
                    found = False
                    continue

