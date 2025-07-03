# Boilerplate implementation of a basic TicTacToe game

class TicTacToe:
    def __init__(self):
        self.board = [0, 1, 2
                      3, 4, 5,
                      6, 7, 8]
        self.x_player = "X"
        self.o_player = "O"

    def move(self, player, position):
        if self.board[position] == "X" or self.board[position] == "O":
            raise Exception("Cannot move to already taken position")
        else:
            self.board[position] = player

    def full_board(self):
        for i in range(9):
            if self.board[i] == i:
                return False
        return True

    def result(self):
        if self.has_won(self.x_player):
            return "X"
        elif self.has_won(self.o_player):
            return "O"
        elif self.full_board():
            return "Draw"
        else:
            return "None"
    
    def end(self):
        if self.result() == "None":
            return False
        return True

    def has_won(self, player):
        # Vertical wins
        for i in range(3):
            full = 0
            for j in range(i, i+7, 3):
                if self.board[j] == player:
                    full++
            if full == 3:
                return True

        # Horizontal wins
        for i in range(0, 7, 3):
            full = 0
            for j in range(i, i+3):
                if self.board[j] == player:
                    full++
            if full == 3:
                return True

        # Diagonal wins
        full = 0
        for i in range(0, 9, 4):
            if self.board[i] == player:
                full++
        if full == 3:
            return True

        full = 0
        for i in range(2, 7, 2):
            if self.board[i] == player:
                full++
        if full == 3:
            return True
        
        return False
