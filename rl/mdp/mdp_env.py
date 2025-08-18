import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any
import logging

# sets up a logger for the MDP environment
logger = logging.getLogger(__name__)

# defines the type of reward structure used in the MDP
RewardType = Literal["state_action_next_state", "state_action", "state"]

class MDPEnv:
    """
    a base class for a finite Markov Decision Process (MDP) environment.

    this class sets up the basic components of an MDP, including states, actions,
    transition probabilities, and reward functions. It can be used to generate
    random MDP environments for testing and experimentation with reinforcement
    learning algorithms.

    attributes:
        num_states (int): the number of states in the environment (S).
        num_actions (int): the number of actions available to the agent (A).
        reward_type (RewardType): the structure of the reward function.
        rng (np.random.Generator): the random number generator used for the environment.
        P (np.ndarray): the transition probability matrix of shape (A, S, S),
                        where P[a, s, s'] is the probability of transitioning to
                        state s' from state s by taking action a.
        R (np.ndarray): the reward matrix. Its shape depends on `reward_type`.
        current_state (int): the current state of the environment.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_type: RewardType = "state_action_next_state",
        seed: Optional[int] = None,
    ):
        """
        initializes the MDP environment.

        args:
            num_states (int): the number of states.
            num_actions (int): the number of actions.
            reward_type (RewardType): the type of reward function to use.
                - 'state_action_next_state': R(s, a, s')
                - 'state_action': R(s, a)
                - 'state': R(s)
            seed (Optional[int]): a seed for the random number generator.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed)

        # sets up logging
        self._setup_logging()

        self.P: np.ndarray
        self.R: np.ndarray

        logger.info(
            f"MDP environment created with {num_states} states, "
            f"{num_actions} actions, and '{reward_type}' rewards."
        )

        self._generate_env()
        self.current_state = self.reset()

    def _setup_logging(self):
        """
        configures the logging for the environment.
        """
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _generate_env(self):
        """
        generates the transition probabilities and reward function for the MDP.
        """
        logger.info("Generating environment dynamics...")
        self._generate_transition_probabilities()
        self._generate_rewards()
        logger.info("Environment generation complete.")

    def _generate_transition_probabilities(self):
        """
        randomly generates the transition probability matrix P.

        the matrix P has shape (num_actions, num_states, num_states), where
        P[a, s, s'] represents P(s' | s, a). The probabilities are normalized
        such that for each state-action pair (s, a), the sum of probabilities
        over all next states s' is 1.
        """
        # generate random transition probabilities
        self.P = self.rng.random((self.num_actions, self.num_states, self.num_states))
        # normalize to ensure valid probability distributions
        self.P /= self.P.sum(axis=2, keepdims=True)
        logger.debug(f"Transition probability matrix P created with shape {self.P.shape}")

    def _generate_rewards(self):
        """
        randomly generates the reward matrix R based on the specified reward_type.

        the shape of R depends on `reward_type`:
        - 'state_action_next_state': (num_actions, num_states, num_states) for R(s, a, s')
        - 'state_action': (num_actions, num_states) for R(s, a)
        - 'state': (num_states,) for R(s)
        """
        if self.reward_type == "state_action_next_state":
            # rewards R(s, a, s') are dependent on state, action, and next state
            self.R = self.rng.random((self.num_actions, self.num_states, self.num_states))
        elif self.reward_type == "state_action":
            # rewards R(s, a) are dependent on state and action
            self.R = self.rng.random((self.num_actions, self.num_states))
        elif self.reward_type == "state":
            # rewards R(s) are dependent only on the state
            self.R = self.rng.random((self.num_states,))
        else:
            # handle unknown reward type
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        logger.debug(f"Reward matrix R created with shape {self.R.shape}")

    def reset(self) -> int:
        """
        resets the environment to a random starting state.

        returns:
            int: the initial state.
        """
        # set current state to a random state
        self.current_state = self.rng.choice(self.num_states)
        logger.info(f"Environment reset. Initial state: {self.current_state}")
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        takes an action in the environment and observes the outcome.

        args:
            action (int): the action to take.

        returns:
            Tuple[int, float, bool, Dict[str, Any]]: a tuple containing:
                - next_state (int): the next state.
                - reward (float): the reward received.
                - done (bool): whether the episode has terminated (always False for this basic MDP).
                - info (Dict[str, Any]): additional information (empty for this basic MDP).
        """
        logger.debug(f"Taking action {action} in state {self.current_state}")
        # get the transition probabilities for the current state and action
        transition_probs = self.P[action, self.current_state, :]

        # sample the next state based on the transition probabilities
        next_state = self.rng.choice(self.num_states, p=transition_probs)

        # calculate the reward based on the reward type
        if self.reward_type == "state_action_next_state":
            reward = self.R[action, self.current_state, next_state]
        elif self.reward_type == "state_action":
            reward = self.R[action, self.current_state]
        else:  # self.reward_type == "state"
            reward = self.R[self.current_state]

        # update the current state
        self.current_state = next_state

        # in this basic MDP, an episode never terminates on its own
        done = False
        info = {}

        logger.debug(
            f"State: {self.current_state}, Action: {action}, "
            f"Next State: {next_state}, Reward: {reward:.2f}"
        )

        return next_state, reward, done, info
