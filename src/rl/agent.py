"""
Q-Learning Agent for Appointment Optimization.

Implements tabular Q-Learning with epsilon-greedy exploration
for learning optimal reminder policies.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import joblib
from pathlib import Path


class QLearningAgent:
    """
    Tabular Q-Learning agent for appointment reminder optimization.
    
    Learns optimal policy through trial-and-error interaction with
    the appointment environment.
    
    Attributes
    ----------
    n_states : int
        Number of discrete states
    n_actions : int
        Number of possible actions
    Q : np.ndarray
        Q-table of shape (n_states, n_actions)
    alpha : float
        Learning rate
    gamma : float
        Discount factor
    epsilon : float
        Current exploration rate
    
    Example
    -------
    >>> agent = QLearningAgent(n_states=1000, n_actions=3)
    >>> history = agent.train(env, episodes=5000)
    >>> policy = agent.get_policy()
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize Q-Learning agent.
        
        Parameters
        ----------
        n_states : int
            Number of discrete states
        n_actions : int
            Number of actions (default: 3 for none/SMS/call)
        alpha : float
            Learning rate
        gamma : float
            Discount factor for future rewards
        epsilon : float
            Initial exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Epsilon decay per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with small random values
        self.Q = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : int
            Current discrete state
        training : bool
            If True, use epsilon-greedy. If False, use greedy.
        
        Returns
        -------
        int
            Chosen action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Update Q-value for state-action pair.
        
        Parameters
        ----------
        state : int
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Next state
        done : bool
            Whether episode ended
        
        Returns
        -------
        float
            TD error
        """
        # Q-Learning update rule
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
    
    def train(self, env, episodes: int = 5000, verbose: bool = True) -> Dict[str, List]:
        """
        Train agent on environment.
        
        Parameters
        ----------
        env : DiscreteAppointmentEnv
            Training environment
        episodes : int
            Number of episodes to train
        verbose : bool
            Print progress
        
        Returns
        -------
        dict
            Training history with rewards and lengths
        """
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                # Safety limit
                if steps > 100:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record history
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Print progress
            if verbose and (episode + 1) % 500 == 0:
                avg_reward = np.mean(self.episode_rewards[-500:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"   Final avg reward (last 500): {np.mean(self.episode_rewards[-500:]):.2f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths
        }
    
    def get_policy(self) -> np.ndarray:
        """
        Get greedy policy from Q-table.
        
        Returns
        -------
        np.ndarray
            Array of best actions for each state
        """
        return np.argmax(self.Q, axis=1)
    
    def get_action_values(self, state: int) -> np.ndarray:
        """Get Q-values for all actions in state."""
        return self.Q[state].copy()
    
    def evaluate(self, env, episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Parameters
        ----------
        env : DiscreteAppointmentEnv
            Evaluation environment
        episodes : int
            Number of evaluation episodes
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        rewards = []
        show_rates = []
        
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state, training=False)
                state, reward, done, info = env.step(action)
                total_reward += reward
            
            rewards.append(total_reward)
            if 'showed' in info:
                show_rates.append(float(info['showed']))
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'show_rate': np.mean(show_rates) if show_rates else 0.0
        }
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'Q': self.Q,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"✅ Agent saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'QLearningAgent':
        """Load agent from disk."""
        data = joblib.load(path)
        agent = cls(
            n_states=data['n_states'],
            n_actions=data['n_actions'],
            alpha=data['alpha'],
            gamma=data['gamma']
        )
        agent.Q = data['Q']
        agent.epsilon = data['epsilon']
        agent.episode_rewards = data.get('episode_rewards', [])
        print(f"✅ Agent loaded from {path}")
        return agent


class RandomAgent:
    """Baseline agent that takes random actions."""
    
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions
    
    def choose_action(self, state: int, training: bool = False) -> int:
        return np.random.randint(self.n_actions)


class AlwaysSMSAgent:
    """Baseline agent that always sends SMS."""
    
    def choose_action(self, state: int, training: bool = False) -> int:
        return 1  # Always SMS


class NoActionAgent:
    """Baseline agent that never intervenes."""
    
    def choose_action(self, state: int, training: bool = False) -> int:
        return 0  # Never act
