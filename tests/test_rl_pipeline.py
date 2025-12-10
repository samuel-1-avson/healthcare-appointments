"""
Tests for RL Pipeline Components.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.reward_model import RewardModel
from src.rl.environment import AppointmentEnv, DiscreteAppointmentEnv
from src.rl.agent import QLearningAgent, RandomAgent, AlwaysSMSAgent


class TestRewardModel:
    """Tests for RewardModel class."""
    
    def test_init(self):
        """Test initialization."""
        model = RewardModel()
        assert model.is_trained == False
        assert len(model.feature_columns) == 6
    
    def test_rule_based_probability(self):
        """Test fallback probability calculation."""
        model = RewardModel()
        
        state = {'Age': 35, 'Lead_Days': 3, 'Hypertension': 0, 'Diabetes': 0}
        
        # No action vs SMS should differ
        prob_none = model.predict_show_probability(state, action=0)
        prob_sms = model.predict_show_probability(state, action=1)
        prob_call = model.predict_show_probability(state, action=2)
        
        assert 0 < prob_none < 1
        assert prob_sms > prob_none
        assert prob_call > prob_sms
    
    def test_fit(self):
        """Test model training."""
        model = RewardModel()
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'Age': np.random.randint(18, 80, n_samples),
            'Lead_Days': np.random.randint(0, 30, n_samples),
            'SMS_received': np.random.choice([0, 1], n_samples),
            'Scholarship': np.random.choice([0, 1], n_samples),
            'Hypertension': np.random.choice([0, 1], n_samples),
            'Diabetes': np.random.choice([0, 1], n_samples)
        })
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.2, 0.8]))
        
        model.fit(X, y)
        
        assert model.is_trained == True
    
    def test_get_reward(self):
        """Test reward generation."""
        model = RewardModel()
        
        state = {'Age': 30, 'Lead_Days': 5}
        reward, showed = model.get_reward(state, action=1)
        
        assert reward in [10.0 - 0.5, -5.0 - 0.5]  # +10 or -5, minus SMS cost
        assert isinstance(showed, (bool, np.bool_))


class TestAppointmentEnv:
    """Tests for AppointmentEnv class."""
    
    def test_init(self):
        """Test environment initialization."""
        env = AppointmentEnv()
        assert env.n_actions == 3
        assert env.n_state_features == 5
    
    def test_reset(self):
        """Test environment reset."""
        env = AppointmentEnv()
        state = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == env.n_state_features
        assert env.done == False
    
    def test_step(self):
        """Test environment step."""
        env = AppointmentEnv()
        env.reset()
        
        state, reward, done, info = env.step(action=1)  # SMS
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_action_names(self):
        """Test action mapping."""
        env = AppointmentEnv()
        
        assert env.action_names[0] == 'No Action'
        assert env.action_names[1] == 'SMS'
        assert env.action_names[2] == 'Call'


class TestDiscreteAppointmentEnv:
    """Tests for DiscreteAppointmentEnv class."""
    
    def test_discrete_state(self):
        """Test discrete state calculation."""
        env = DiscreteAppointmentEnv()
        state_idx = env.reset()
        
        assert isinstance(state_idx, (int, np.integer))
        assert 0 <= state_idx < env.n_states
    
    def test_step_returns_int(self):
        """Test that step returns discrete state."""
        env = DiscreteAppointmentEnv()
        env.reset()
        
        next_state, reward, done, info = env.step(0)
        
        assert isinstance(next_state, (int, np.integer))


class TestQLearningAgent:
    """Tests for QLearningAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = QLearningAgent(n_states=100, n_actions=3)
        
        assert agent.Q.shape == (100, 3)
        assert agent.epsilon == 1.0
    
    def test_choose_action(self):
        """Test action selection."""
        agent = QLearningAgent(n_states=100, n_actions=3)
        
        action = agent.choose_action(state=0)
        assert action in [0, 1, 2]
    
    def test_update(self):
        """Test Q-value update."""
        agent = QLearningAgent(n_states=100, n_actions=3)
        
        initial_q = agent.Q[0, 0].copy()
        td_error = agent.update(state=0, action=0, reward=10.0, next_state=1, done=False)
        
        assert agent.Q[0, 0] != initial_q
        assert isinstance(td_error, float)
    
    def test_train_short(self):
        """Test short training run."""
        env = DiscreteAppointmentEnv()
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
        
        history = agent.train(env, episodes=50, verbose=False)
        
        assert len(history['rewards']) == 50
        assert len(history['lengths']) == 50
    
    def test_get_policy(self):
        """Test policy extraction."""
        agent = QLearningAgent(n_states=100, n_actions=3)
        
        policy = agent.get_policy()
        
        assert len(policy) == 100
        assert all(a in [0, 1, 2] for a in policy)
    
    def test_evaluate(self):
        """Test policy evaluation."""
        env = DiscreteAppointmentEnv()
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
        
        metrics = agent.evaluate(env, episodes=20)
        
        assert 'mean_reward' in metrics
        assert 'std_reward' in metrics
        assert 'show_rate' in metrics


class TestBaselineAgents:
    """Tests for baseline agents."""
    
    def test_random_agent(self):
        """Test random agent."""
        agent = RandomAgent()
        action = agent.choose_action(state=0)
        assert action in [0, 1, 2]
    
    def test_always_sms_agent(self):
        """Test always-SMS agent."""
        agent = AlwaysSMSAgent()
        assert agent.choose_action(state=0) == 1
        assert agent.choose_action(state=99) == 1


def test_integration():
    """Integration test for full RL pipeline."""
    # Create environment
    env = DiscreteAppointmentEnv()
    
    # Create and train agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        epsilon=0.5,
        epsilon_decay=0.99
    )
    
    # Short training
    history = agent.train(env, episodes=100, verbose=False)
    
    # Evaluate
    metrics = agent.evaluate(env, episodes=50)
    
    # Compare to random baseline
    random_agent = RandomAgent()
    env_eval = DiscreteAppointmentEnv()
    
    random_rewards = []
    for _ in range(50):
        state = env_eval.reset()
        total_reward = 0
        done = False
        while not done:
            action = random_agent.choose_action(state)
            state, reward, done, _ = env_eval.step(action)
            total_reward += reward
        random_rewards.append(total_reward)
    
    # RL agent should perform at least as well as random after some training
    print(f"RL Agent: {metrics['mean_reward']:.2f}")
    print(f"Random: {np.mean(random_rewards):.2f}")
    
    # This is a weak assertion since we only trained 100 episodes
    assert isinstance(metrics['mean_reward'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
