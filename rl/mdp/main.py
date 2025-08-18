import logging
from mdp_env import MDPEnv

def main():
    """
    a simple example of how to use the MDPEnv class.

    this function initializes an MDP environment, runs a random agent for a few
    time steps, and prints the interactions.
    """
    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # parameters for the MDP environment
    num_states = 5
    num_actions = 2
    reward_type = 'state_action'
    seed = 42

    # create an instance of the MDP environment
    env = MDPEnv(
        num_states=num_states,
        num_actions=num_actions,
        reward_type=reward_type,
        seed=seed
    )

    # reset the environment to get the initial state
    current_state = env.reset()
    logging.info(f"Initial state: {current_state}")

    # run the simulation for a few steps with a random agent
    num_steps = 10
    for t in range(num_steps):
        # the agent randomly selects an action
        action = env.rng.choice(env.num_actions)

        # the agent takes the action in the environment
        next_state, reward, done, info = env.step(action)

        # print the interaction details
        logging.info(
            f"Time step {t + 1}: "
            f"State={current_state}, Action={action}, "
            f"Next State={next_state}, Reward={reward:.2f}"
        )

        # update the current state
        current_state = next_state

if __name__ == "__main__":
    main() 