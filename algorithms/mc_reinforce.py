import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64]):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        if hidden_layers[0] != 0:
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x)


class MCREINFORCE:
    def __init__(
        self,
        env,
        use_baseline=False,
        seed=0,
        hidden_layers=[64, 64],
        learning_rate=1e-3,
    ):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.use_baseline = use_baseline

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, hidden_layers=hidden_layers
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim)
            self.value_optimizer = optim.Adam(
                self.value_net.parameters(), lr=learning_rate
            )

    def train(self, episodes=1000, gamma=0.99):
        rewards_per_episode = []
        for episode in tqdm(range(episodes), desc="Training MC REINFORCE"):
            state = self.env.reset()[0]
            states, actions, rewards = [], [], []
            done = False

            while not done:
                action_probs = self.policy_net(torch.FloatTensor(state))
                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # Calculate returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)

            # Update policy
            states = np.array(states)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            action_probs = (
                self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            )

            if self.use_baseline:
                values = self.value_net(states).squeeze()
                advantages = returns - values
                policy_loss = -(torch.log(action_probs) * advantages.detach()).mean()
                value_loss = nn.MSELoss()(values, returns)
            else:
                policy_loss = -(torch.log(action_probs) * returns).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if self.use_baseline:
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

            rewards_per_episode.append(sum(rewards))

        return rewards_per_episode

    def save_model(self, model_name):
        model_path = os.path.join(f"{model_name}.pth")
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"MC REINFORCE model saved to {model_path}\n")

    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
