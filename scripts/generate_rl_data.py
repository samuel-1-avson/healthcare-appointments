"""
Generate Synthetic RL Dataset.

Creates a dataset of (State, Action, Reward, Next_State) transitions
for offline RL training and analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.rl.reward_model import RewardModel
from src.rl.environment import AppointmentEnv


def generate_rl_episodes(
    n_episodes: int = 10000,
    max_steps: int = 10,
    reward_model: RewardModel = None
) -> pd.DataFrame:
    """
    Generate RL episodes with (S, A, R, S') transitions.
    
    Parameters
    ----------
    n_episodes : int
        Number of episodes to generate
    max_steps : int
        Maximum steps per episode
    reward_model : RewardModel
        Trained reward model (uses default if None)
    
    Returns
    -------
    pd.DataFrame
        Dataset with columns: episode, step, state_*, action, reward, next_state_*, done
    """
    env = AppointmentEnv(reward_model=reward_model or RewardModel(), max_days=7)
    
    records = []
    
    for episode in tqdm(range(n_episodes), desc="Generating episodes"):
        state_vec = env.reset()
        state_dict = env.current_state.copy()
        
        for step in range(max_steps):
            # Random policy for exploration
            action = np.random.randint(env.n_actions)
            
            # Take step
            next_state_vec, reward, done, info = env.step(action)
            next_state_dict = env.current_state.copy()
            
            # Record transition
            records.append({
                # Episode info
                'episode': episode,
                'step': step,
                
                # State features
                'state_days_left': env.days_left + 1,  # Before step
                'state_age': state_dict.get('Age', 35),
                'state_chronic': int(state_dict.get('Hypertension', 0) or 
                                     state_dict.get('Diabetes', 0)),
                'state_prev_noshow_rate': state_dict.get('prev_noshow_rate', 0.2),
                'state_interventions': env.interventions_sent - (1 if action > 0 else 0),
                'state_sms_received': state_dict.get('SMS_received', 0),
                
                # Action
                'action': action,
                'action_name': env.action_names[action],
                
                # Reward
                'reward': reward,
                
                # Next state
                'next_state_days_left': env.days_left,
                'next_state_interventions': env.interventions_sent,
                
                # Episode info
                'done': done,
                'showed': info.get('showed', None) if done else None
            })
            
            if done:
                break
            
            # Update for next iteration
            state_vec = next_state_vec
            state_dict = next_state_dict.copy()
    
    df = pd.DataFrame(records)
    
    print(f"\n✅ Generated {len(df):,} transitions from {n_episodes:,} episodes")
    print(f"   Avg steps per episode: {len(df) / n_episodes:.1f}")
    print(f"   Action distribution: {df['action_name'].value_counts().to_dict()}")
    
    return df


def main():
    print("=" * 60)
    print("GENERATING RL DATASET")
    print("=" * 60)
    
    # Generate dataset
    df = generate_rl_episodes(n_episodes=10000, max_steps=8)
    
    # Save to CSV
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "rl_episodes.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal transitions: {len(df):,}")
    print(f"Total episodes: {df['episode'].nunique():,}")
    
    # Reward distribution
    print(f"\nReward statistics:")
    print(f"  Mean: {df['reward'].mean():.3f}")
    print(f"  Std: {df['reward'].std():.3f}")
    print(f"  Min: {df['reward'].min():.3f}")
    print(f"  Max: {df['reward'].max():.3f}")
    
    # Show rate (completed episodes)
    completed = df[df['done'] == True]
    if len(completed) > 0:
        show_rate = completed['showed'].mean()
        print(f"\nShow rate: {show_rate:.2%}")
    
    # Sample data
    print("\n📋 Sample transitions:")
    print(df.head(10).to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
