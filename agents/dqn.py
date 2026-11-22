import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.9, epsilon=0.2):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer = ReplayBuffer()
        self.update_target_interval = 5
        self.step_count = 0
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return 0
        with torch.no_grad():
            q_vals = self.model(torch.tensor(state, dtype=torch.float32))
            return int(torch.argmax(q_vals).item())
    def learn(self, batch_size=2):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next = self.target_model(next_states).max(1)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.update_target_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
