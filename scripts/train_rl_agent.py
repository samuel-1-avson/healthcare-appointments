"""
Training Script for RL Appointment Optimization Agent.

Trains the Q-Learning agent and compares against baseline policies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path

from src.rl.reward_model import RewardModel
from src.rl.environment import DiscreteAppointmentEnv
from src.rl.agent import QLearningAgent, RandomAgent, AlwaysSMSAgent, NoActionAgent


def load_training_data(db_path: str = "healthcare.db") -> tuple:
    """Load training data from database."""
    # Try multiple paths
    paths = [db_path, f"data/{db_path}", f"../{db_path}"]
    conn = None
    
    for p in paths:
        if os.path.exists(p):
            conn = sqlite3.connect(p)
            break
    
    if conn is None:
        print("⚠️ Database not found, using simulated data")
        return None, None
    
    df = pd.read_sql_query("SELECT * FROM appointments", conn)
    conn.close()
    
    # Prepare features and target
    feature_cols = ['Age', 'Lead_Days', 'SMS_received', 'Scholarship', 'Hypertension', 'Diabetes']
    
    # Handle missing Lead_Days
    if 'Lead_Days' not in df.columns:
        df['Lead_Days'] = (pd.to_datetime(df['AppointmentDay']) - 
                           pd.to_datetime(df['ScheduledDay'])).dt.days.clip(lower=0)
    
    X = df[feature_cols].fillna(0)
    y = df['No_Show'].apply(lambda x: 0 if x == 1 else 1)  # Invert: 1 = showed
    
    return X, y


def train_reward_model(X, y) -> RewardModel:
    """Train reward model on historical data."""
    reward_model = RewardModel()
    
    if X is not None:
        reward_model.fit(X, y)
    else:
        print("Using rule-based reward model")
    
    return reward_model


def evaluate_agent(agent, env, n_episodes: int = 500) -> dict:
    """Evaluate agent performance."""
    return agent.evaluate(env, episodes=n_episodes)


def compare_policies(env, agents: dict, n_episodes: int = 500) -> pd.DataFrame:
    """Compare multiple policies."""
    results = []
    
    for name, agent in agents.items():
        metrics = evaluate_agent(agent, env, n_episodes)
        results.append({
            'Policy': name,
            'Mean Reward': metrics['mean_reward'],
            'Std Reward': metrics['std_reward'],
            'Show Rate': metrics['show_rate']
        })
    
    return pd.DataFrame(results)


def plot_training_progress(rewards: list, window: int = 100, save_path: str = None):
    """Plot training reward progress."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    smoothed = pd.Series(rewards).rolling(window).mean()
    ax.plot(smoothed, color='#3b82f6', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Total Reward ({window}-episode avg)')
    ax.set_title('Q-Learning Training Progress', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training plot saved to {save_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("RL AGENT TRAINING: Appointment Optimization")
    print("=" * 60)
    
    # Load data and train reward model
    print("\n📊 Loading training data...")
    X, y = load_training_data()
    
    print("\n🎯 Training reward model...")
    reward_model = train_reward_model(X, y)
    
    # Create environment
    print("\n🌍 Creating RL environment...")
    env = DiscreteAppointmentEnv(reward_model=reward_model, max_days=7)
    print(f"   State space: {env.n_states} states")
    print(f"   Action space: {env.n_actions} actions")
    
    # Train Q-Learning agent
    print("\n🤖 Training Q-Learning agent...")
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    history = agent.train(env, episodes=5000, verbose=True)
    
    # Compare with baselines
    print("\n📈 Comparing with baseline policies...")
    agents = {
        'Q-Learning': agent,
        'Random': RandomAgent(),
        'Always SMS': AlwaysSMSAgent(),
        'No Action': NoActionAgent()
    }
    
    comparison = compare_policies(env, agents, n_episodes=500)
    print("\n" + "=" * 60)
    print("POLICY COMPARISON RESULTS")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    # Save agent
    save_dir = Path("models/rl")
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(save_dir / "q_learning_agent.joblib")
    reward_model.save(save_dir / "reward_model.joblib")
    
    # Save training plot
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_training_progress(history['rewards'], save_path=fig_dir / "rl_training_progress.png")
    
    # Print optimal policy
    print("\n" + "=" * 60)
    print("LEARNED OPTIMAL POLICY (Sample States)")
    print("=" * 60)
    
    action_names = ['No Action', 'SMS', 'Call']
    print("\nAction mapping for sample day ranges:")
    for days in [7, 5, 3, 1, 0]:
        # Sample states for different days
        sample_state = days * (env.age_bins * env.chronic_bins * env.noshow_bins * env.intervention_bins)
        if sample_state < env.n_states:
            best_action = np.argmax(agent.Q[sample_state])
            print(f"  Day {days}: {action_names[best_action]}")
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {save_dir}")
    print(f"   Plot saved to: {fig_dir}")


if __name__ == "__main__":
    main()
