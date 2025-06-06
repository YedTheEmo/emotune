import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from .parameter_space import MusicParameterSpace
from emotune.utils.logging import get_logger

logger = get_logger()

class SoftActorCritic:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Temperature parameter for entropy regularization
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim
        
        # Hyperparameters
        self.tau = 0.005  # Soft update rate
        self.gamma = 0.99  # Discount factor
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]
    
    def update(self, batch: Dict[str, torch.Tensor]):
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update critics
        self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        self._update_actor(states)
        
        # Update temperature
        self._update_temperature(states)
        
        # Soft update target networks
        self._soft_update_targets()
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
    
    def _update_actor(self, states):
        actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def _update_temperature(self, states):
        actions, log_probs = self.actor.sample(states)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
    def _soft_update_targets(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            
            
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            
            
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.action_scale = 0.1  # Limit action magnitude for smooth transitions
        self.log_std_min = -10
        self.log_std_max = 2
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
    def get_deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, param_space: MusicParameterSpace):
        self.param_space = param_space
        self.param_names = list(param_space.parameters.keys())
        self.action_dim = len(self.param_names)
        
        # State: [emotion_mean (2), emotion_cov (3), dtw_error (1), trajectory_progress (1)]
        self.state_dim = 7
        
        self.sac = SoftActorCritic(self.state_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Current parameter values
        self.current_params = param_space.get_default_parameters()
            
    def get_state_vector(self, emotion_mean: np.ndarray, emotion_cov: list, 
                        dtw_error: float, trajectory_progress: float) -> np.ndarray:
        """Convert system state to RL state vector"""
        cov_flat = np.array([emotion_cov[0][0], emotion_cov[1][1], emotion_cov[0][1]])
        
        state = np.concatenate([
            np.asarray(emotion_mean).reshape(-1),
            cov_flat.reshape(-1),
            np.array([dtw_error]),
            np.array([trajectory_progress])
        ])
        
        return state
   
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Dict[str, float]:
        """Select parameter adjustments based on current state"""
        action = self.sac.select_action(state, deterministic)
        
        # Convert action to parameter adjustments
        param_adjustments = {}
        for i, param_name in enumerate(self.param_names):
            
            
            param_adjustments[param_name] = action[i]
        
        return param_adjustments
    
    def update_parameters(self, adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply adjustments to current parameters"""
        for param_name, adjustment in adjustments.items():
            
            
            if param_name in self.current_params:
                self.current_params[param_name] += adjustment
        
        # Clip to valid ranges
        self.current_params = self.param_space.clip_parameters(self.current_params)
        return self.current_params.copy()
    
    def store_transition(self, state: np.ndarray, action: Dict[str, float], 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        # Convert action dict to array
        action_array = np.array([action.get(param, 0.0) for param in self.param_names])
        
        self.replay_buffer.push(state, action_array, reward, next_state, done)
    
    def store_experience(self, state: np.ndarray, emotion: dict, deviation: float, session_time: float):
        """Store experience for RL training. This is a wrapper for store_transition."""
        # For now, we use a placeholder reward and next_state, done
        # In a real system, reward and next_state should be computed based on feedback and environment
        action = self.select_action(state)
        reward = -abs(deviation)  # Example: negative deviation as reward
        next_state = state  # Placeholder: in practice, should be the next observed state
        done = False  # Placeholder: set True if session ends
        self.store_transition(state, action, reward, next_state, done)

    def get_training_summary(self) -> dict:
        """Return a summary of RL training status."""
        return {
            'buffer_size': len(self.replay_buffer),
            'current_params': self.current_params.copy(),
        }
    
    def train(self, batch_size: int = 256):
        """Train the RL agent if enough data is available"""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        self.sac.update(batch)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            'next_states': torch.FloatTensor(np.array(next_states)),
            'dones': torch.FloatTensor(np.array(dones)).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)

