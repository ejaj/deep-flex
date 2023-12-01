class TicTacToe:
    def __init__(self):
        # Initialize an empty 3x3 board with spaces
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        # Start with 'X' as the first player
        self.current_player = 'X'

    def print_board(self):
        # Print the current state of the board
        for row in self.board:
            print(" ".join(row))

    def check_winner(self, player):
        # Check if the current player has won
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)) or all(self.board[j][i] == player for j in range(3)):
                return True

        if all(self.board[i][i] == player for i in range(3)) or all(self.board[i][2 - i] == player for i in range(3)):
            return True

        return False

    def is_board_full(self):
        # Check if the board is full
        return all(all(cell != ' ' for cell in row) for row in self.board)

    def make_move(self, row, col):
        # Make a move at the specified row and column
        if self.board[row][col] == ' ':
            # If the cell is empty, place the current player's symbol
            self.board[row][col] = self.current_player

            # Check for a winner or a tie
            if self.check_winner(self.current_player):
                self.print_board()
                print(f"Player {self.current_player} wins!")
                return True
            elif self.is_board_full():
                self.print_board()
                print("It's a tie!")
                return True

            # Switch to the other player for the next turn
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            print("Invalid move. Cell already taken. Try again.")


# Create an instance of the TicTacToe class
game = TicTacToe()

# Game loop
while True:
    # Display the current state of the board
    game.print_board()

    # Get player input for the row and column
    row = int(input(f"Player {game.current_player}, enter row (0, 1, or 2): "))
    col = int(input(f"Player {game.current_player}, enter column (0, 1, or 2): "))

    # Make the move and check if the game is over
    if game.make_move(row, col):
        break
