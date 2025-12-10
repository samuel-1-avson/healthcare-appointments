"""
Reward Model for RL Environment.

Trains on historical appointment data to predict probability of patient showing up
given their features and the intervention action taken.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class RewardModel:
    """
    Reward predictor trained on historical appointment data.
    
    Predicts P(show | state, action) using a RandomForest classifier,
    which is then used by the RL environment to generate rewards.
    
    Attributes
    ----------
    model : RandomForestClassifier
        Trained classifier for show/no-show prediction
    scaler : StandardScaler
        Feature scaler for normalization
    feature_columns : list
        Names of feature columns used for training
    is_trained : bool
        Whether the model has been trained
    
    Example
    -------
    >>> reward_model = RewardModel()
    >>> reward_model.fit(X_train, y_train)
    >>> prob = reward_model.predict_show_probability(state, action)
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize RewardModel.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = ['Age', 'Lead_Days', 'SMS_received', 'Scholarship', 
                                 'Hypertension', 'Diabetes']
        self.is_trained = False
        self._action_effect = {
            0: 0.0,    # No action
            1: 0.08,   # SMS boost
            2: 0.15    # Call boost
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RewardModel':
        """
        Train the reward model on historical data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with columns matching feature_columns
        y : pd.Series
            Binary target (0 = no-show, 1 = showed)
        
        Returns
        -------
        RewardModel
            Self for method chaining
        """
        # Ensure correct columns
        X_train = X[self.feature_columns].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"✅ RewardModel trained on {len(X)} samples")
        print(f"   Feature importance: {dict(zip(self.feature_columns, 
              [f'{x:.3f}' for x in self.model.feature_importances_]))}")
        
        return self
    
    def predict_show_probability(self, state: Dict[str, Any], action: int) -> float:
        """
        Predict probability of patient showing up given state and action.
        
        Parameters
        ----------
        state : dict
            Patient state with keys matching feature_columns
        action : int
            Action taken (0=none, 1=SMS, 2=call)
        
        Returns
        -------
        float
            Probability of showing up [0, 1]
        """
        if not self.is_trained:
            # Fallback to rule-based if not trained
            return self._rule_based_probability(state, action)
        
        # Create feature vector
        features = np.array([[
            state.get('Age', 35),
            state.get('Lead_Days', 7),
            1 if action >= 1 else state.get('SMS_received', 0),  # SMS if action
            state.get('Scholarship', 0),
            state.get('Hypertension', 0),
            state.get('Diabetes', 0)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        base_prob = self.model.predict_proba(features_scaled)[0][1]
        
        # Apply action effect
        action_boost = self._action_effect.get(action, 0)
        final_prob = min(0.95, base_prob + action_boost)
        
        return final_prob
    
    def _rule_based_probability(self, state: Dict[str, Any], action: int) -> float:
        """Fallback probability calculation when model not trained."""
        base_prob = 0.80
        
        # Age effect
        age = state.get('Age', 35)
        if age < 25:
            base_prob -= 0.05
        elif age > 60:
            base_prob += 0.05
        
        # Lead time effect
        lead_days = state.get('Lead_Days', 7)
        base_prob -= 0.01 * lead_days
        
        # Chronic conditions help (more invested in health)
        if state.get('Hypertension', 0) or state.get('Diabetes', 0):
            base_prob += 0.05
        
        # Action effect
        base_prob += self._action_effect.get(action, 0)
        
        return max(0.1, min(0.95, base_prob))
    
    def get_reward(self, state: Dict[str, Any], action: int, step_cost: float = 0.1) -> Tuple[float, bool]:
        """
        Get reward for state-action pair.
        
        Parameters
        ----------
        state : dict
            Current patient state
        action : int
            Action taken
        step_cost : float
            Cost per step/intervention
        
        Returns
        -------
        Tuple[float, bool]
            (reward, showed_up)
        """
        show_prob = self.predict_show_probability(state, action)
        showed_up = np.random.random() < show_prob
        
        # Reward structure
        if showed_up:
            reward = 10.0  # Patient attended
        else:
            reward = -5.0  # No-show
        
        # Subtract intervention cost
        if action == 1:
            reward -= 0.5   # SMS cost
        elif action == 2:
            reward -= 1.0   # Call cost
        
        return reward, showed_up
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }, path)
        print(f"✅ RewardModel saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RewardModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_columns = data['feature_columns']
        instance.is_trained = data['is_trained']
        print(f"✅ RewardModel loaded from {path}")
        return instance
