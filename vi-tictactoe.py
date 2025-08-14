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

    previous_turn = False
    turn_info = [] # [tuple: init-state, int: action, tuple: dest-state, int: immediate reward]
                   # should refresh only every player turn (not while loop iter)
                   # using previous_turn bool
    while game.end() == False:
        state = game.board[:]
        state.append(player)
        state = tuple(state)
        found = False
        if game.turn() == player:
            if previous_turn == True:
                # need to append dest-state, immediate reward to turn_info
                # and add info to dictionaries rt, tt, & vt
                turn_info.append(state) # append dest-state
                # find out way to append immediate reward if game is over smh
            turn_info = []
            turn_info.append(state) # append init_state
            possible_actions = game.possible()
            for p in possible_actions:
                state_plus = state + (p,)
                if state_plus in tt.keys():
                    found = True # do more edits: choose & append action to turn_info
            if found == False:
                action = random.choice(possible_actions)
                turn_info.append(action)
            previous_turn = True
