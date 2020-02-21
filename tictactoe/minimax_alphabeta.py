import random

class Game:
    X = 'X'
    O = '0'
    EMPTY = ' '
    TIE = 0
    NUM_SQUARES = 9
    TREE_DEPTH = 9
    MAX, MIN = 1000, -1000

    def __init__(self):
        self.board = self.new_board()        

    def display_instruct(self):
        """Display game instructions."""
        print(
        """
        Welcome to the greatest intellectual challenge of all time: Tic-Tac-Toe. This will be a showdown between your human brain and my silicon processor.
        
        You will make your move known by entering a number, 0 - 8. The number will correspond to the board position as illustrated:
        0 | 1 | 2 
        ---------
        3 | 4 | 5 
        --------- 
        6 | 7 | 8
        Prepare yourself, human. The ultimate battle is about to begin. \n """
        )

    def new_board(self):
        """Create new game board.""" 
        board = []
        for square in range(self.NUM_SQUARES):
            board.append(self.EMPTY) 
        return board

    def display_board(self, board):
        """Display game board on screen."""
        print("\n\t", board[0], "|", board[1], "|", board[2]) 
        print("\t", "---------")
        print("\t", board[3], "|", board[4], "|", board[5]) 
        print("\t", "---------")
        print("\t", board[6], "|", board[7], "|", board[8], "\n")

    def legal_moves(self, board):
        """Create list of legal moves."""
        moves = []
        for square in range(self.NUM_SQUARES):
            if board[square] == self.EMPTY: 
                moves.append(square)
        return moves

    def winner(self, board):
        """Determine the game winner."""
        WAYS_TO_WIN = ((0, 1, 2),
                        (3, 4, 5),
                        (6, 7, 8),
                        (0, 3, 6),
                        (1, 4, 7),
                        (2, 5, 8),
                        (0, 4, 8),
                        (2, 4, 6))
        
        for row in WAYS_TO_WIN:
            if board[row[0]] == board[row[1]] == board[row[2]] != self.EMPTY:
                winner = board[row[0]]
                return winner

        if self.EMPTY not in board:
            return self.TIE
    
    def ask_yes_no(self, question):
        """ask a yes or now question"""
        response = None
        while response not in ("y", "n"):
            response = input(question).lower()
        return response

    def ask_number(self, question, low, high):
        """Ask for a number within a range.""" 
        response = None
        while response not in range(low, high):
            try:
                response = int(input(question)) 
            except ValueError:
                pass
        return response

    def pieces(self):
        """Determine if player or computer goes first."""
        go_first = self.ask_yes_no("Do you require the first move? (y/n): ") 
        if go_first == "y":
            print("\nThen take the first move. You will need it.") 
            human = self.X
            computer = self.O
        else:
            print("\nYour bravery will be your undoing... I will go first.") 
            computer = self.X
            human = self.O
        return computer, human

    def next_turn(self, turn):
        """Switch turns"""
        if turn == self.X:
            return self.O
        else:
            return self.X

    def human_move(self):
        """get human move"""

        legal = self.legal_moves(self.board)
        move = None

        while move not in legal:
            move = self.ask_number("Make a move? (0 - 8):", 0, self.NUM_SQUARES)
            if move not in legal:
                print("Can't move there")
        print("Fine....")

        return move
   
    def check_terminal_state(self, board, computer):
        winner = self.winner(board)
        if winner is not None: 
            if winner == self.TIE:
                return True, 0
            elif winner == computer:
                return True, 1
            else: 
                return True, -1
        else:
            return False, None
    
    def is_board_empty(self, board):        
        for space in board:
            if space != self.EMPTY:
                return False
        return True

    def computer_opening_move(self, board):
        if self.is_board_empty(board):
            corner_moves = [0,2,6,8]
            return corner_moves[random.randrange(0,3)]

        return -1

    def minimax(self, computer, maximizing_player, turn, board, depth):
        """            
        created one function to handle entier mimimax algorithm, instead of having a starter function
        """
        # beginning game moves
        opening_move = self.computer_opening_move(board)
        if opening_move > -1:
            return opening_move

        # get current legal moves
        legal = self.legal_moves(board)  
        
        # terminal state - check for end of game move
        # check for winner or tie
        terminal, state = self.check_terminal_state(board, computer)
        if terminal:
            return state
        
        if depth > self.TREE_DEPTH or len(legal) == 0:
            return 0  
        
        # holds results for current board moves
        minimax_list = []  
        
        # analyze all legal moves and append results to minimax_list
        for move in legal:
            new_board = board.copy()
            new_board[move] = turn
             
            if maximizing_player:  
                result = self.minimax(computer, False, self.X, new_board, depth+1 )
                if result is not None:                    
                    minimax_list.append(result)
            else:
                result = self.minimax(computer, True, self.O, new_board, depth+1 )
                if result is not None:                    
                    minimax_list.append(result)

        # if back at top of tree return the best move
        if depth == 0:
            # print(f'max player {max(minimax_list)}')
            return minimax_list.index(max(minimax_list))
        else:
            # return the max or min of the minimax_list 
            if maximizing_player:
                return max(minimax_list)
            else:
                return min(minimax_list)
    
    def minimax_decision_aima(self, computer, turn, board, depth):
        """
        version of minimax that is closest to AIMA pseudocode, has a starter function            
        """
        opening_move = self.computer_opening_move(board)
        if opening_move > -1:
            return opening_move
            
        moves = self.legal_moves(board)  
        best_scores = []  
        for move in moves:     
            new_board = board.copy()
            new_board[move] = turn
            best_score = self.minimax_aima(computer, False, self.O, new_board, depth )
            best_scores.append(best_score)

        print(f'best_scores {best_scores}')
        
        return best_scores.index(max(best_scores))        
        
    def minimax_aima(self, computer, maximizing_player, turn, board, depth):
        """
        version of minimax that is closest to AIMA pseudocode,
        """
        # get current legal moves
        legal = self.legal_moves(board)  
        
        # terminal state - check for end of game move
        # check for winner or tie
        terminal, state = self.check_terminal_state(board, computer)
        if terminal:
            return state
            
        # depth check
        if depth > self.TREE_DEPTH or len(legal) == 0:
            return 0  
        
        if maximizing_player:
            v = self.MIN
        else:
            v = self.MAX
        
        # analyze all legal moves and append results to minimax_list
        for move in legal:            
            new_board = board.copy()
            new_board[move] = turn          
             
            if maximizing_player:                                             
                v = max(v, self.minimax_aima(computer, False, self.O, new_board, depth+1))
            else:                
                v = min(v, self.minimax_aima(computer, True, self.X, new_board, depth+1 ))
       
        return v

    def minimax_alpha(self, computer, maximizing_player, turn, board, depth, alpha, beta):
        """
        minimax alpha beta pruning algorithm.
        """
        # check for opening move 
        opening_move = self.computer_opening_move(board)
        if opening_move > -1:
            return opening_move
        
        # check terminal state
        terminal, state = self.check_terminal_state(board, computer)
        if terminal:
            return state, 0

        legal = self.legal_moves(board)  

        if depth > self.TREE_DEPTH or len(legal) == 0:
            return 0, 0
        
        if maximizing_player:  
            best = self.MIN
            best_index = 0
            for index, move in enumerate(legal):                                                
                new_board = board.copy()
                new_board[move] = turn
                
                result, not_used = self.minimax_alpha(computer, False, self.next_turn(turn), new_board, depth+1, alpha, beta)
                if depth == 0:
                    print(result)
                if result > best:
                    best_index = index   
                
                best = max(best, result)

                # prune tree
                if best >= beta:                    
                    return best, best_index

                alpha = max(alpha, best)     
                
            return best, best_index                
        else:           
            best = self.MAX     
            best_index = 0
            for index, move in enumerate(legal):                
                new_board = board.copy()
                new_board[move] = turn
                result, not_used = self.minimax_alpha(computer, True, self.next_turn(turn), new_board, depth+1, alpha, beta)

                if result < best:
                    best_index = index   
                
                best = min(best, result)

                # prune tree
                if best <= alpha:
                    return best, best_index
                
                beta = min(beta, best)
                
            return best, best_index
        
    def winner_message(self, human):
        winner = game.winner(game.board)
        if winner == human:
            print('The human won')
        elif winner == 0:
            print('it was a tie')
        else:
            print('the computer won')

    def run(self):
        self.display_instruct()
        computer, human = self.pieces()
        turn = self.X
        winner = None
        
        while winner is None:
            if turn == human:
                move = self.human_move()
                self.board[move] = human
            else:         
                ### change AI algorithm here       
                # alpha beta                
                # not_used, move_index = self.minimax_alpha(computer, True, computer, self.board, 0, self.MIN, self.MAX)               
                
                #mini max based of aima book
                move_index = self.minimax_decision_aima(computer, computer, self.board, 0)

                # mini max combined into one function
                # move_index = self.minimax(computer, True, computer, self.board, 0)
                
                # print(f'min final {move_index}')
                
                legal = self.legal_moves(self.board)

                self.board[legal[move_index]] = computer
                
            self.display_board(self.board)
            winner = self.winner(self.board)
            turn = self.next_turn(turn)
            # print(self.winner(self.board))
        self.winner_message(human)

if __name__ == '__main__':
    game = Game()
    game.run()