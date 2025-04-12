import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import os


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64], type=1):
        super(DuelingDQN, self).__init__()
        self.type = type
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        self.value = nn.Sequential(nn.Linear(input_dim, 1))
        self.advantage = nn.Sequential(nn.Linear(input_dim, action_dim))

    def forward(self, x):
        features = self.feature(x)
        values = self.value(features)
        advantages = self.advantage(features)
        if self.type == 1:
            adv_mean = advantages.mean(dim=1, keepdim=True)
        else:
            adv_mean = advantages.max(dim=1, keepdim=True)[0]
        return values + (advantages - adv_mean)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size=64):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, env, dueling_type=1, seed=0, hidden_layers=[64, 64], learning_rate=1e-4
    ):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.dueling_type = dueling_type

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.q_net = DuelingDQN(
            self.state_dim,
            self.action_dim,
            hidden_layers=hidden_layers,
            type=dueling_type,
        )
        self.target_q_net = DuelingDQN(
            self.state_dim,
            self.action_dim,
            hidden_layers=hidden_layers,
            type=dueling_type,
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()

        self.epsilon = 1.0

    def update(self):
        if len(self.memory) < 64:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        with torch.no_grad():
            next_q = self.target_q_net(next_states)
            next_q_val = next_q.max(1)[0]
            target = rewards + 0.99 * next_q_val * (1 - dones)

        q_values = self.q_net(states)
        action_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(action_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        rewards_per_episode = []
        for episode in tqdm(range(episodes), desc="Training Dueling DQN"):
            state = self.env.reset()[0]
            total_reward = 0
            for t in range(500):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = (
                            self.q_net(torch.FloatTensor(state).unsqueeze(0))
                            .argmax()
                            .item()
                        )

                next_state, reward, done, _, _ = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                self.update()

                state = next_state
                total_reward += reward
                if done:
                    break

            rewards_per_episode.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)

            if episode % 10 == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        return rewards_per_episode

    def save_model(self, model_name):
        model_path = os.path.join(f"{model_name}.pth")
        torch.save(self.q_net.state_dict(), model_path)
        print(f"Dueling DQN model saved to {model_path}\n")

    def load_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))
        self.q_net.eval()
