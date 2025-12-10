"""
Gym-Compatible Appointment Environment for RL.

Provides a simulation environment for training RL agents to optimize
appointment reminder strategies.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .reward_model import RewardModel


class AppointmentEnv:
    """
    Gym-compatible environment for appointment reminder optimization.
    
    The agent learns when and how to intervene (SMS/call) to maximize
    patient attendance while minimizing intervention costs.
    
    State Space (5 dimensions):
    - days_until_appt: int [0-14]
    - age_normalized: float [0-1]
    - chronic_condition: int {0, 1}
    - prev_noshow_rate: float [0-1]
    - interventions_sent: int [0-5]
    
    Action Space (3 actions):
    - 0: No action
    - 1: Send SMS
    - 2: Make call
    
    Reward Structure:
    - +10: Patient shows up
    - -5: Patient no-shows
    - -0.5: SMS sent (cost)
    - -1.0: Call made (cost)
    
    Attributes
    ----------
    reward_model : RewardModel
        Model for predicting show probability
    max_days : int
        Maximum days until appointment
    max_interventions : int
        Maximum interventions allowed
    """
    
    def __init__(self, reward_model: Optional[RewardModel] = None, max_days: int = 7):
        """
        Initialize environment.
        
        Parameters
        ----------
        reward_model : RewardModel, optional
            Trained reward model. Uses rule-based if None.
        max_days : int
            Maximum days until appointment at start
        """
        self.reward_model = reward_model or RewardModel()
        self.max_days = max_days
        self.max_interventions = 5
        
        # State variables
        self.current_state = None
        self.days_left = 0
        self.interventions_sent = 0
        self.done = False
        
        # Action and state dimensions
        self.n_actions = 3
        self.n_state_features = 5
    
    def reset(self, patient_features: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Parameters
        ----------
        patient_features : dict, optional
            Specific patient features. Random if None.
        
        Returns
        -------
        np.ndarray
            Initial state vector
        """
        if patient_features is None:
            patient_features = self._generate_random_patient()
        
        self.current_state = patient_features
        self.days_left = np.random.randint(3, self.max_days + 1)
        self.interventions_sent = 0
        self.done = False
        
        return self._get_state_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Parameters
        ----------
        action : int
            Action to take (0=none, 1=SMS, 2=call)
        
        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state_vector(), 0.0, True, {}
        
        # Update state based on action
        if action in [1, 2]:
            self.interventions_sent += 1
            self.current_state['SMS_received'] = 1
        
        self.current_state['Lead_Days'] = self.days_left
        
        # Check if appointment day
        if self.days_left <= 0:
            # Get final reward
            reward, showed = self.reward_model.get_reward(self.current_state, action)
            self.done = True
            info = {'showed': showed, 'interventions': self.interventions_sent}
            return self._get_state_vector(), reward, True, info
        
        # Intermediate step
        reward = 0.0
        if action == 1:
            reward -= 0.5  # SMS cost
        elif action == 2:
            reward -= 1.0  # Call cost
        
        # Advance time
        self.days_left -= 1
        
        return self._get_state_vector(), reward, False, {}
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state to normalized vector."""
        return np.array([
            self.days_left / self.max_days,
            self.current_state.get('Age', 35) / 100,
            float(self.current_state.get('Hypertension', 0) or 
                  self.current_state.get('Diabetes', 0)),
            self.current_state.get('prev_noshow_rate', 0.2),
            self.interventions_sent / self.max_interventions
        ], dtype=np.float32)
    
    def _generate_random_patient(self) -> Dict[str, Any]:
        """Generate random patient features."""
        return {
            'Age': np.random.randint(18, 80),
            'Lead_Days': self.max_days,
            'SMS_received': 0,
            'Scholarship': np.random.choice([0, 1], p=[0.9, 0.1]),
            'Hypertension': np.random.choice([0, 1], p=[0.8, 0.2]),
            'Diabetes': np.random.choice([0, 1], p=[0.93, 0.07]),
            'prev_noshow_rate': np.random.beta(2, 8)
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return {
            'days_left': self.days_left,
            'patient': self.current_state,
            'interventions': self.interventions_sent,
            'done': self.done
        }
    
    @property
    def action_names(self) -> Dict[int, str]:
        """Map action indices to names."""
        return {0: 'No Action', 1: 'SMS', 2: 'Call'}


class DiscreteAppointmentEnv(AppointmentEnv):
    """
    Discrete state version for tabular Q-Learning.
    
    Discretizes continuous state into bins for Q-table lookup.
    """
    
    def __init__(self, reward_model: Optional[RewardModel] = None, max_days: int = 7):
        super().__init__(reward_model, max_days)
        
        # Discretization bins
        self.days_bins = max_days + 1  # 0 to max_days
        self.age_bins = 4               # Young, Adult, Middle, Senior
        self.chronic_bins = 2           # Yes/No
        self.noshow_bins = 3            # Low, Medium, High
        self.intervention_bins = 3      # 0, 1-2, 3+
        
        # Total states
        self.n_states = (self.days_bins * self.age_bins * self.chronic_bins * 
                         self.noshow_bins * self.intervention_bins)
    
    def get_discrete_state(self) -> int:
        """Convert current state to discrete index."""
        state_vec = self._get_state_vector()
        
        # Discretize each dimension
        days_idx = min(int(state_vec[0] * self.days_bins), self.days_bins - 1)
        age_idx = min(int(state_vec[1] * self.age_bins), self.age_bins - 1)
        chronic_idx = int(state_vec[2])
        noshow_idx = min(int(state_vec[3] * self.noshow_bins), self.noshow_bins - 1)
        interv_idx = min(int(state_vec[4] * self.intervention_bins), self.intervention_bins - 1)
        
        # Combine into single index
        state_idx = (days_idx * self.age_bins * self.chronic_bins * self.noshow_bins * self.intervention_bins +
                     age_idx * self.chronic_bins * self.noshow_bins * self.intervention_bins +
                     chronic_idx * self.noshow_bins * self.intervention_bins +
                     noshow_idx * self.intervention_bins +
                     interv_idx)
        
        return state_idx
    
    def reset(self, patient_features: Optional[Dict[str, Any]] = None) -> int:
        """Reset and return discrete state."""
        super().reset(patient_features)
        return self.get_discrete_state()
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Step and return discrete state."""
        _, reward, done, info = super().step(action)
        return self.get_discrete_state(), reward, done, info
