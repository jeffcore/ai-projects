class NQueens():
    """
        Board is represented as a 1D array
        Example: 4x4 board
        [3,2,0,1]    
        ROWS: array indexes are rows    [Row 0, Row 1, Row 2, Row 3]
        COLUMNS: the integer is the column index  [3,2,0,1]   3 is in row 0 column 3
        [1,3,0,2]
        would create the folowing board
        0 Q 0 0 
        0 0 0 Q 
        Q 0 0 0 
        0 0 Q 0 
        
        Why do it this way? Yes it is a little confusing but it makes for easier valid position checks

        Some Hints:
        Valid Position Check:
            * you do not have to check if placement in same row is valid - only one queen can be in a row      
            * you only have to check previous rows
            * diagonals you only have to check diagonal left UP and right UP, not DOWN  

    """
    def __init__(self, n=4):
       self.result = []
       self.n = n
       self.col_placement = [None] * n     
    
    def run(self):
        self.solver_n_queens(0)

    def solver_n_queens(self, row):
        if row == self.n:           
            self.result.append(list(self.col_placement))
            return
        
        for col in range(self.n):
            self.col_placement[row] = col
            if (self.is_valid(row)):
                # recursion only if it is a valid move other wise all is forgotten
                self.solver_n_queens(row+1)
            
    def is_valid(self, row):
        """
        row is the current row placement we are verifying
            these checks are easier using a few match tricks because of the 1D array board representation
            otherwise you would have to loop through all positions on a 2D array board represenation
        """
        for i in range(row):
            # check previous columns
            if self.col_placement[i] == self.col_placement[row]:
                return False
            # check diagonal left UP    if position is 3 previous position cant be [1,2,3,NONE]   
            elif self.col_placement[i] == self.col_placement[row] - (row-i):
                return False
            # check diagonal right UP if position is 1 previous position cant be [1,2,3,NONE]   
            elif self.col_placement[i] + i == row + self.col_placement[row]:
                return False
             
        return True
    
    def build_board(self, arr):
        board = []
        n = len(arr)
        for i in range(self.n):
            row = []
            for k in range(self.n):
                if k == arr[i]:
                    row.append('Q')
                else:
                    row.append('_')
            board.append(row)            
        return board   
    
    def print_board(self, board):
        for row in board:
            for col in row:
                print(col, end=' ')
            print(' ')
        print('\n')
    
    def print_results(self):    
        print(f'Number of Solutions: {len(self.result)}')    
        for col_place in self.result:
            board = self.build_board(col_place)
            self.print_board(board)
        
def main():        
    n_queens = NQueens(4)
    n_queens.run()
    n_queens.print_results()


if __name__ == "__main__":
    main()