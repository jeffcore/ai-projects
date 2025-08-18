import random
import copy

class TicTacToe:
    """
    Represents the Tic-Tac-Toe game environment.
    Board: 0 = empty, 1 = Player X, -1 = Player O.
    """
    def __init__(self):
        self.board = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) # Initial empty board
        self.current_player = 1 # Player X starts

    def get_empty_cells(self, board_state):
        """Returns a list of (row, col) tuples for empty cells."""
        empty_cells = []
        for r in range(3):
            for c in range(3):
                if board_state[r][c] == 0:
                    empty_cells.append((r, c))
        return empty_cells

    def make_move(self, board_state, move, player):
        """Returns a new board state after applying the move."""
        row, col = move
        # For training, assume valid moves are chosen from get_empty_cells.
        if not (0 <= row < 3 and 0 <= col < 3) or board_state[row][col] != 0:
            raise ValueError(f"Invalid move {move} attempted on board {board_state} by player {player}")
        
        new_board_list = [list(r) for r in board_state]
        new_board_list[row][col] = player
        return tuple(tuple(r) for r in new_board_list)

    def check_win(self, board_state, player):
        """Checks if the given player has won."""
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(board_state[i][j] == player for j in range(3)): return True # Row i
            if all(board_state[j][i] == player for j in range(3)): return True # Column i
        
        # Check diagonals
        if board_state[0][0] == player and board_state[1][1] == player and board_state[2][2] == player:
            return True
        if board_state[0][2] == player and board_state[1][1] == player and board_state[2][0] == player:
            return True
            
        return False

    def is_game_over(self, board_state):
        """Checks if the game has ended (win or draw)."""
        return self.check_win(board_state, 1) or \
               self.check_win(board_state, -1) or \
               not self.get_empty_cells(board_state) # Board full

    def get_winner(self, board_state):
        """Returns the ID of the winning player (1 for X, -1 for O, 0 for draw, None if game not over)."""
        if self.check_win(board_state, 1):
            return 1  # X wins
        if self.check_win(board_state, -1):
            return -1 # O wins
        if not self.get_empty_cells(board_state):
            return 0  # Draw
        return None # Game not over

    def get_canonical_form(self, board_state):
        """
        Returns a canonical representation of the board to handle symmetries.
        The source mentions symmetry as an exercise [7] rather than a core part of the basic algorithm.
        For this implementation, we will skip full canonicalization for simplicity and
        return the board state itself.
        """
        return board_state

class RLAgent:
    """
    The reinforcement learning agent for Tic-Tac-Toe (plays as Player X).
    It learns a state-value function over 'afterstates' (board positions after its move)
    using a Temporal-Difference (TD(0)) update rule.
    """
    def __init__(self, game_env, alpha=0.1, epsilon=0.1):
        self.game = game_env
        self.alpha = alpha  # Step-size parameter [4, 8]
        self.epsilon = epsilon  # Exploration parameter [9]
        # Value function: maps canonical board states (afterstates) to their estimated value [3]
        # The key is a board_tuple, value is a float (estimated probability of winning for Player X).
        self.values = {}

    def get_value(self, board_state):
        """
        Retrieves the estimated value of a board state from the value function.
        New states are initialized with an initial guess of 0.5 [3].
        Terminal states (win/loss/draw for X) are explicitly set to their true values (1.0 or 0.0) [3].
        """
        canonical_state = self.game.get_canonical_form(board_state)
        
        # Check if it's a terminal state and assign its true value as per the problem description [3]
        winner = self.game.get_winner(canonical_state)
        if winner is not None:
            if winner == 1:  # X wins
                self.values[canonical_state] = 1.0
            else:  # O wins or Draw
                self.values[canonical_state] = 0.0
        
        # For non-terminal states or if not already explicitly set
        if canonical_state not in self.values:
            self.values[canonical_state] = 0.5  # Initial guess for unknown non-terminal states [3]
            
        return self.values[canonical_state]

    def choose_action(self, current_board):
        """
        Chooses an action (move) for the agent (Player X) using an epsilon-greedy strategy.
        It returns the chosen move and a boolean indicating if it was a greedy (non-exploratory) move.
        
        [9]: "Most of the time we move greedily... Occasionally, however, we select randomly"
        [8]: "Exploratory moves do not result in any learning"
        """
        available_moves = self.game.get_empty_cells(current_board)
        if not available_moves:
            return None, False # No moves possible
        
        if random.random() < self.epsilon:
            # Exploratory move: select randomly [9]
            return random.choice(available_moves), False # False indicates it's an exploratory move
        else:
            # Greedy move: select the move that leads to the state with the greatest value [9]
            best_projected_value = -float('inf') # For Player X, higher value is better
            best_moves_list = [] # To handle random tie-breaking among equally good moves [10]
            
            for move in available_moves:
                # Simulate agent's move to get the 'afterstate' (St in TD update)
                potential_agent_afterstate = self.game.make_move(current_board, move, 1)
                
                if self.game.is_game_over(potential_agent_afterstate):
                    # If game ends after agent's move, its value is the final outcome for X.
                    winner = self.game.get_winner(potential_agent_afterstate)
                    value_from_this_move = 1.0 if winner == 1 else 0.0
                else:
                    # If game continues, estimate the value of 'potential_agent_afterstate' (St).
                    # This value depends on the opponent's subsequent move. The example assumes an
                    # imperfect opponent, so we model this as an average over the values of all
                    # possible states resulting from random opponent responses.
                    
                    temp_opponent_moves = self.game.get_empty_cells(potential_agent_afterstate)
                    if not temp_opponent_moves: # Should not happen if game not over and moves available
                        value_from_this_move = 0.5 # Default fallback
                    else:
                        sum_next_afterstate_values = 0
                        for op_move_sim in temp_opponent_moves:
                            # Simulate opponent's random move. This gives St+1 (after opponent's move).
                            temp_board_after_O = self.game.make_move(potential_agent_afterstate, op_move_sim, -1)
                            sum_next_afterstate_values += self.get_value(temp_board_after_O)
                        value_from_this_move = sum_next_afterstate_values / len(temp_opponent_moves)
                
                # Update best move(s) if this move is better or equally good
                if value_from_this_move > best_projected_value:
                    best_projected_value = value_from_this_move
                    best_moves_list = [move] 
                elif value_from_this_move == best_projected_value:
                    best_moves_list.append(move)
            
            # Pick one from the best moves (random tie-breaking)
            return random.choice(best_moves_list), True # True indicates it's a greedy move

    def update_value(self, state_St, state_St_plus_1):
        """
        Performs the Temporal-Difference (TD(0)) update on the value function.
        [8]: V(St) <- V(St) + alpha * (V(St+1) - V(St))
        
        state_St: The 'earlier state' (an afterstate, board after agent's greedy move).
        state_St_plus_1: The 'later state' (an afterstate, board after opponent's subsequent move from state_St).
        """
        # Get the current estimated value of the state to be updated
        value_St = self.get_value(state_St) # Ensures state_St is in self.values with a value
        
        # Determine the target for the update (Rt+1 + gamma * V(St+1))
        # For Tic-Tac-Toe, rewards are implicitly 0 until game end.
        # If state_St_plus_1 is terminal, its value is its true fixed outcome (1.0 or 0.0).
        # Otherwise, bootstrap from its current estimated value.
        
        # The `get_value` method already handles setting terminal states to their true values.
        # So, we can directly use `self.get_value(state_St_plus_1)` as the target.
        target_value = self.get_value(state_St_plus_1)
        
        # Perform the TD(0) update
        canonical_St = self.game.get_canonical_form(state_St)
        self.values[canonical_St] = value_St + self.alpha * (target_value - value_St)

def play_game(agent, opponent_player_id=-1):
    """
    Simulates a single game between the RL agent (Player X) and a random opponent (Player O).
    It applies TD updates online, as implied by the source's description [11].
    """
    game_env = TicTacToe()
    current_board_state = game_env.board
    
    # This variable holds the agent's 'afterstate' (St) from its last *greedy* move,
    # waiting for the opponent's response to become St+1 for an update.
    last_greedy_agent_afterstate = None 

    # Game loop continues until the game is over
    while not game_env.is_game_over(current_board_state):
        if game_env.current_player == 1: # Agent's turn (Player X)
            # Agent chooses a move using its learned policy
            chosen_X_move, was_greedy_action = agent.choose_action(current_board_state)
            
            if chosen_X_move is None: # Should only happen if game is already over
                break
            
            # Execute agent's chosen move, resulting in a new board state (the 'afterstate', St)
            board_after_X_move = game_env.make_move(current_board_state, chosen_X_move, 1)
            
            # If the agent's move was greedy, we store this 'afterstate' (St).
            # We will wait for the opponent's response to get St+1 and perform the TD update.
            if was_greedy_action:
                last_greedy_agent_afterstate = board_after_X_move
            else:
                # If the move was exploratory, no learning happens for this specific move [8].
                # Clear any pending greedy afterstate from a previous turn.
                last_greedy_agent_afterstate = None 
                
            current_board_state = board_after_X_move # Update the current board state
        
        else: # Opponent's turn (Player O)
            opponent_moves = game_env.get_empty_cells(current_board_state)
            if opponent_moves:
                # Opponent makes a random move, as described for an imperfect player [2]
                opponent_chosen_move = random.choice(opponent_moves)
                current_board_state = game_env.make_move(current_board_state, opponent_chosen_move, -1)
            
            # After the opponent's move, `current_board_state` is now St+1 (the 'later state').
            # If there was a pending greedy agent afterstate (last_greedy_agent_afterstate, which is St),
            # we can now perform the TD update.
            if last_greedy_agent_afterstate is not None:
                # Apply the temporal-difference update for the agent's previous greedy move
                agent.update_value(last_greedy_agent_afterstate, current_board_state)
                # Once updated, this 'afterstate' is no longer pending.
                last_greedy_agent_afterstate = None 
            
        game_env.current_player *= -1 # Switch player for the next turn

    # The game has ended.
    # If the agent made a greedy move right before the game ended (e.g., agent wins),
    # there might be a final pending update where the terminal state is St+1.
    if last_greedy_agent_afterstate is not None:
        agent.update_value(last_greedy_agent_afterstate, current_board_state) 
    
    # Return the winner of the game
    return game_env.get_winner(current_board_state)

def train_agent(agent, num_episodes):
    """
    Trains the RL agent by playing a specified number of episodes.
    """
    win_count_X = 0
    draw_count = 0
    loss_count_X = 0
    
    print(f"**Training agent for {num_episodes} episodes...**")
    for episode in range(1, num_episodes + 1):
        winner = play_game(agent) # Play one game
        if winner == 1:
            win_count_X += 1
        elif winner == 0:
            draw_count += 1
        else: # winner == -1 (O wins)
            loss_count_X += 1
        
        # Print progress periodically
        if episode % (num_episodes // 10 if num_episodes >= 10 else 1) == 0:
            print(f"Episode {episode}/{num_episodes} - Wins (X): {win_count_X}, Draws: {draw_count}, Losses (X): {loss_count_X}")
    
    print("\n**Training complete.**")
    print(f"Final results against random opponent: **Wins (X): {win_count_X}, Draws: {draw_count}, Losses (X): {loss_count_X}**")
    print(f"Number of unique afterstates learned: {len(agent.values)}")
    
    # Example of some learned values (for player X, who is learning)
    print("\n**Some learned afterstate values (for Player X):**")
    # An empty board after X makes its first move to (0,0)
    initial_move_board = ((1,0,0),(0,0,0),(0,0,0)) 
    print(f"  Board with X at (0,0) (after X's first move): {agent.get_value(initial_move_board):.4f}")

    # A board where X is about to win (X's turn, next move will win)
    # E.g., X at (0,0), (0,1), next move at (0,2) wins.
    nearly_winning_board_for_X = ((1,1,0), (0,-1,0), (0,0,0)) # X is at (0,0), (0,1), O at (1,1)
    # If it's X's turn to move from this state and X can make it a win, its value should be high.
    # The agent's value function is for *afterstates* (after X has moved).
    # So we should query for a board that X just moved to, which *now* has a high value.
    
    # Example: X wins by placing at (0,2)
    winning_afterstate_example = ((1,1,1),(0,-1,0),(0,0,0))
    print(f"  Winning state for X (X has completed row): {agent.get_value(winning_afterstate_example):.4f}")

    # Example: O wins, so value for X is 0
    losing_afterstate_example = ((-1,-1,-1),(1,1,0),(0,0,0))
    print(f"  Losing state for X (O has completed row): {agent.get_value(losing_afterstate_example):.4f}")


# Main execution
if __name__ == "__main__":
    env = TicTacToe()
    # Parameters for the RL agent. Alpha is the learning rate, epsilon is for exploration.
    # The source states that performance converges well if alpha is reduced over time [4],
    # but for simplicity, we use a fixed small alpha.
    rl_agent = RLAgent(env, alpha=0.01, epsilon=0.1) # alpha=0.01, epsilon=0.1 are common starting points
    
    # Train for a sufficient number of episodes. Tic-Tac-Toe has a relatively small state space.
    # 100,000 games should be enough for the agent to learn a good policy against a random opponent.
    train_agent(rl_agent, num_episodes=100000)

    play_game(rl_agent, opponent_player_id=1)
