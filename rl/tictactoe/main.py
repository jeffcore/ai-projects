import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board
        self.current_winner = None
        logging.info("TicTacToe game initialized.")

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            logging.debug(f"Making move: Player {letter} places on square {square}")
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
                logging.info(f"Player {letter} wins!")
            return True
        logging.warning(f"Invalid move attempt: Square {square} already taken.")
        return False

    def winner(self, square, letter):
        # Check rows, columns, and diagonals
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in column]):
            return True
        # Check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0,4,8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2,4,6]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

    def is_full(self):
        full = ' ' not in self.board
        if full:
            logging.info("Board is full.")
        return full

    def reset(self):
        logging.info("Resetting the game board.")
        self.board = [' '] * 9
        self.current_winner = None



import random
import pickle

class QLearningAgent:
    """represents an agent that learns to play TicTacToe using Q-learning.

    attributes:
        q_table (dict): stores the Q-values for state-action pairs. keys are (state, action), values are Q-values.
        alpha (float): the learning rate, determining how much new information overrides old information.
        gamma (float): the discount factor, valuing future rewards.
        epsilon (float): the exploration rate, determining the probability of choosing a random action.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        """initializes the QLearningAgent.

        args:
            alpha (float): learning rate.
            gamma (float): discount factor.
            epsilon (float): exploration rate.
        """
        self.q_table = {}  # state-action values
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        logging.info(f"QLearningAgent initialized with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")

    def get_state(self, board):
        """converts the game board list into an immutable tuple to be used as a state key.

        args:
            board (list): the current state of the tic-tac-toe board.

        returns:
            tuple: the board state represented as a tuple.
        """
        return tuple(board)

    def choose_action(self, state, available_moves):
        """chooses an action based on the epsilon-greedy strategy.

        with probability epsilon, chooses a random action from available moves (exploration).
        with probability 1-epsilon, chooses the action with the highest Q-value for the current state (exploitation).
        if multiple actions have the same max Q-value, one is chosen randomly.

        args:
            state (tuple): the current state of the game.
            available_moves (list): a list of possible moves (indices) from the current state.

        returns:
            int: the chosen action (square index).
        """
        if not available_moves:
            logging.warning("choose_action called with no available moves.")
            return None # Or handle this case as appropriate
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(available_moves)
            logging.debug(f"Exploring: Chose random action {action}")
        else:
            q_values = {a: self.q_table.get((state, a), 0) for a in available_moves}
            max_q = max(q_values.values())
            # Handle the case where q_values might be empty if available_moves was empty, though checked above
            max_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(max_actions) # Choose randomly among best actions
            logging.debug(f"Exploiting: Chose action {action} with Q-value {max_q}")
        return action

    def learn(self, state, action, reward, next_state, done, next_available_moves):
        """updates the Q-value for the taken state-action pair using the Q-learning rule.

        q(s, a) = q(s, a) + alpha * (reward + gamma * max(q(s', a')) - q(s, a))
        where s is the current state, a is the action taken, s' is the next state,
        and max(q(s', a')) is the maximum Q-value for the next state over all possible next actions.

        args:
            state (tuple): the state before the action was taken.
            action (int): the action taken.
            reward (float): the reward received after taking the action.
            next_state (tuple): the state after the action was taken.
            done (bool): True if the game ended after the action, False otherwise.
            next_available_moves (list): list of available moves in the next_state.
            
        returns:
            tuple: a tuple containing (old_q, target, new_q, max_future_q) for logging/debugging purposes.
        """
        old_q = self.q_table.get((state, action), 0)
        max_future_q = 0 # Initialize to 0, will be updated if not done
        
        if done:
            target = reward # if game is done, the future value is just the reward obtained
        else:
            # calculate the maximum Q-value for the next state
            if not next_available_moves:
                max_future_q = 0 # if no moves available from next state, future reward is 0
            else:
                future_q_values = [self.q_table.get((next_state, a), 0) for a in next_available_moves]
                max_future_q = max(future_q_values)
            
            target = reward + self.gamma * (-max_future_q) # use negative of opponent's best outcome
            
        # q-learning update rule
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(state, action)] = new_q
        # log the learning step internally
        logging.debug(f"Learning Internal: State={state}, Action={action}, Reward={reward}, NextState={next_state}, Done={done}, OldQ={old_q:.4f}, Target={target:.4f}, NewQ={new_q:.4f}")
        
        # return calculated values for external logging/debugging
        return old_q, target, new_q, max_future_q # Added max_future_q to return

    def save_policy(self, file_name):
        """saves the learned Q-table to a file using pickle.

        args:
            file_name (str): the name of the file to save the policy to.
        """
        logging.info(f"Saving Q-table policy to {file_name}")
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self.q_table, f)
            logging.info("Policy saved successfully.")
        except Exception as e:
             logging.error(f"Error saving policy to {file_name}: {e}")

    def load_policy(self, file_name):
        """loads a Q-table from a file using pickle.

        if the file is not found, it logs a warning and continues with an empty Q-table.

        args:
            file_name (str): the name of the file to load the policy from.
        """
        logging.info(f"Loading Q-table policy from {file_name}")
        try:
            with open(file_name, 'rb') as f:
                self.q_table = pickle.load(f)
            logging.info("Policy loaded successfully.")
        except FileNotFoundError:
            logging.warning(f"Policy file {file_name} not found. Starting with an empty Q-table.")
            self.q_table = {} # ensure q_table is reset if file not found
        except Exception as e:
            logging.error(f"Error loading policy file {file_name}: {e}")
            self.q_table = {} # ensure q_table is reset on other errors


def train(agent, episodes=10000, debug_mode=False):
    """trains the QLearningAgent by playing episodes of TicTacToe.

    args:
        agent (QLearningAgent): the agent to train.
        episodes (int): the number of games to play for training.
        debug_mode (bool): if True, prints detailed step info and pauses after each step.
    """
    logging.info(f"Starting training for {episodes} episodes. Debug mode: {debug_mode}")
    game = TicTacToe()
    for episode in range(episodes):
        if not debug_mode and (episode + 1) % 1000 == 0: # Log progress periodically only if not in debug
             logging.info(f"Completed episode {episode + 1}/{episodes}")
        elif debug_mode:
             logging.debug(f"Starting Episode {episode + 1}/{episodes}") # Use debug for less clutter if main log is INFO

        game.reset()
        state = agent.get_state(game.board)
        done = False
        current_player = 'X'
        turn = 0
        while not done:
            turn += 1
            # store state before action for learning and debugging
            prev_state_tuple = state
            prev_state_board = list(prev_state_tuple) # for printing board

            available_moves = game.available_moves()
            if not available_moves:
                # this state occurs if the board is full ending in a draw, but the last player's move didn't win.
                # the loop should have ended in the previous iteration if there was a winner.
                logging.info(f"Episode {episode + 1}, Turn {turn}: No available moves. Should be a draw.")
                # the game should already be marked as done by is_full() in the previous step's check.
                # ensure the loop terminates correctly, maybe break here if needed, though `done` should be True.
                if not game.is_full():
                     logging.warning(f"Episode {episode + 1}, Turn {turn}: No moves but board not full? State: {state}")
                break

            action = agent.choose_action(state, available_moves)
            prev_action = action # store action for learning
            
            # make the move
            if not game.make_move(action, current_player):
                 logging.error(f"Episode {episode + 1}, Turn {turn}: Agent chose an invalid move {action}. State: {state}, Available: {available_moves}. Skipping turn.")
                 # Skip learning for this turn as the state didn't change meaningfully due to invalid move attempt.
                 # It might be better to penalize the agent here, but for now, just log and switch player.
                 current_player = 'O' if current_player == 'X' else 'X' # Switch player even on invalid move to prevent potential infinite loops
                 continue

            # get the result of the action
            next_state = agent.get_state(game.board)
            next_available_moves = game.available_moves()
            reward = 0
            if game.current_winner == current_player:
                reward = 1 # Winner gets positive reward
                done = True
            elif game.is_full():
                reward = 0 # Draw reward is 0
                done = True
            # Note: A reward of 0 is implicitly given for non-terminal moves.

            # learn from the move
            # The reward and next_state are from the perspective of the current_player
            old_q, target, new_q, max_future_q = agent.learn(prev_state_tuple, prev_action, reward, next_state, done, next_available_moves)

            # ---- Debug Step Output ----
            if debug_mode:
                print("\n" + "-"*40)
                print(f"Episode: {episode + 1}, Turn: {turn}, Player: {current_player}")
                print("Agent Settings:")
                print(f"  alpha  (Learning Rate): {agent.alpha}")
                print(f"  gamma  (Discount Factor): {agent.gamma}")
                print(f"  epsilon (Exploration Rate): {agent.epsilon}")
                print("\nPrevious State (s):")
                print(format_board_str(prev_state_board))
                print(f"\nAction Taken (a): {prev_action}")
                print(f"Reward Received (r) for Player {current_player}'s move: {reward}") # Clarify reward context
                print("\nNext State (s'):")
                print(format_board_str(game.board))
                print(f"Game Done after move? {done}")
                if game.current_winner:
                    print(f"Winner: {game.current_winner}")
                elif game.is_full():
                    print("Result: Draw")
                
                print("\nQ-Value Update Calculation (for Player " + current_player + "'s move: " + str(prev_action) + "):") # Clarify update context
                print(f"  Q(s, a)_old = {old_q:.4f}")
                # Explain target calculation
                if done:
                    # If game ended, target is just the reward for this final move
                    print(f"  Target = reward = {reward:.4f} (Game ended)")
                else:
                    # If game continues, target incorporates the discounted value of the opponent's best response
                    # max_future_q is the best Q-value the *next* player (opponent) can get from next_state
                    print(f"  Max Q-Value for Opponent from Next State (max Q(s', a')) = {max_future_q:.4f}")
                    print(f"  Target = reward + gamma * (-max Q(s', a'))") # Negate opponent's best outcome
                    print(f"         = {reward} + {agent.gamma} * (-{max_future_q:.4f})" )
                    print(f"         = {target:.4f}")
                print(f"  New Q(s, a) = Q_old + alpha * (Target - Q_old)")
                print(f"              = {old_q:.4f} + {agent.alpha} * ({target:.4f} - {old_q:.4f})")
                print(f"              = {new_q:.4f}")
                print("-"*40)
                
                # interactive pause
                while True:
                    user_cmd = input("Press Enter for next step, or type 'q' to print Q-table: ").strip().lower()
                    if user_cmd == 'q':
                        print("\nCurrent Q-Table States:")
                        if not agent.q_table:
                            print("  (empty)")
                        else:
                            # Sort items for potentially more consistent viewing, though dict order isn't guaranteed pre 3.7
                            sorted_items = sorted(agent.q_table.items(), key=lambda item: item[0]) 
                            for (state_key, action_key), q_value in sorted_items:
                                print(f"  State: {state_key}, Action: {action_key} -> Q-Value: {q_value:.4f}")
                        print("-"*20)
                        # Prompt again after printing
                        continue 
                    elif user_cmd == '':
                        # Enter pressed, continue to next step
                        break
                    else:
                        print("Invalid command.")

            # ---- End Debug Step Output ----

            # move to the next state for the next iteration
            state = next_state 
            # switch player only if the game is not done
            if not done:
                current_player = 'O' if current_player == 'X' else 'X'

    logging.info("Training finished.")

def format_board_str(board):
    """formats the board list into a multi-line string representation,
       showing the index number for empty squares.
    """
    s = ""
    for i in range(0, 9, 3):
        # If the spot is empty (' '), show the index number. Otherwise, show the player mark ('X' or 'O').
        cell1 = ' ' if board[i]   == ' ' else board[i]
        cell2 = ' ' if board[i+1] == ' ' else board[i+1]
        cell3 = ' ' if board[i+2] == ' ' else board[i+2]
        
        s += f" {cell1} | {cell2} | {cell3} \n"
        if i < 6:
            s += "---|---|---\n"
    return s.strip() # remove trailing newline

def play_game(agent):
    """allows a human player to play against the trained agent."""
    game = TicTacToe()
    human_player = 'X' # Human plays X
    agent_player = 'O'  # Agent plays O
    current_player = human_player

    # ensure agent uses learned policy without exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    logging.info(f"Starting game against agent. Human is {human_player}, Agent is {agent_player}. Agent epsilon set to 0.")

    # Print reference board
    print("\nBoard Squares (0-8):")
    print(" 0 | 1 | 2 ")
    print("---|---|---")
    print(" 3 | 4 | 5 ")
    print("---|---|---")
    print(" 6 | 7 | 8 \n")

    done = False
    while not done:
        print("\nCurrent Board:")
        print_board(game.board) # This will print the board with 'X' and 'O'

        if current_player == human_player:
            available_moves = game.available_moves()
            valid_move = False
            while not valid_move:
                try:
                    move_str = input(f"Your turn ({human_player}). Enter square (0-8): ")
                    square = int(move_str)
                    if square in available_moves:
                        valid_move = True
                    else:
                        print(f"Invalid move. Choose from available squares: {available_moves}")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 8.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Optionally re-raise or break if it's critical
            
            if game.make_move(square, human_player):
                logging.info(f"Human ({human_player}) played square {square}")
            else:
                 # This case should ideally not be reached due to validation loop
                logging.error("Human attempted invalid move despite validation.")
                continue # Should not happen, but just in case

        else: # Agent's turn
            state = agent.get_state(game.board)
            available_moves = game.available_moves()
            if not available_moves:
                logging.info("Agent's turn, but no moves available. Game ends?")
                break # Should be caught by is_full or winner check later
            
            action = agent.choose_action(state, available_moves) # Epsilon is 0, so it chooses best action
            if game.make_move(action, agent_player):
                logging.info(f"Agent ({agent_player}) played square {action}")
                print(f"Agent ({agent_player}) plays square {action}")
            else:
                logging.error(f"Agent attempted invalid move {action}. State: {state}, Available: {available_moves}")
                # This indicates a bug in agent logic or game state if it happens
                break

        # check for game end
        if game.current_winner:
            print("\nFinal Board:")
            print_board(game.board)
            winner_msg = f"Player {game.current_winner} wins!"
            print(winner_msg)
            logging.info(winner_msg)
            done = True
        elif game.is_full():
            print("\nFinal Board:")
            print_board(game.board)
            draw_msg = "It's a draw!"
            print(draw_msg)
            logging.info(draw_msg)
            done = True
        else:
            # switch player
            current_player = agent_player if current_player == human_player else human_player

    # restore agent's original epsilon if needed elsewhere
    agent.epsilon = original_epsilon
    logging.info("Game finished.")

def print_board(board):
    """prints the tic-tac-toe board to the console."""
    # use the string formatting function
    print(format_board_str(board))

if __name__ == "__main__":
    agent = QLearningAgent()
    policy_file = "tictactoe_q_policy.pkl"

    # add flag for detailed step-by-step debugging
    debug_steps = input("Enable step-by-step debugging for training? (yes/no): ").lower() == 'yes'

    mode = input("Choose mode: (1) Train (2) Play against agent: ")

    if mode == '1':
        logging.info("Mode selected: Training")
        # Optional: Load pre-trained policy to continue training
        try:
            agent.load_policy(policy_file)
        except FileNotFoundError:
            logging.info(f"No pre-trained policy file '{policy_file}' found, starting fresh training.")
        except Exception as e:
             logging.error(f"Error loading policy file '{policy_file}' for training: {e}")

        # pass the debug flag to the train function
        train(agent, episodes=100000, debug_mode=debug_steps) 
        agent.save_policy(policy_file)

    elif mode == '2':
        logging.info("Mode selected: Play against agent")
        try:
            agent.load_policy(policy_file)
            play_game(agent)
        except FileNotFoundError:
            logging.error(f"Policy file '{policy_file}' not found. Cannot play. Please train the agent first (mode 1).")
            print(f"Error: Policy file '{policy_file}' not found. Train the agent first.")
        except Exception as e:
            logging.error(f"Error loading policy file '{policy_file}' for playing: {e}")
            print(f"An error occurred loading the policy: {e}")
    else:
        print("Invalid mode selected. Please enter '1' or '2'.")
        logging.warning(f"Invalid mode selected: {mode}")

    logging.info("Script finished.")
