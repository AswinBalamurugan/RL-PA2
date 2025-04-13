import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os


class BaseNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers=[64, 64]):
        self.hidden_layers = hidden_layers
        super(BaseNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        if hidden_layers != [0]:
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x):
        return self.features(x)


class PolicyNetwork(BaseNetwork):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64]):
        super(PolicyNetwork, self).__init__(state_dim, hidden_layers)
        self.policy_head = nn.Sequential(
            nn.Linear(self.input_dim, action_dim), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        features = super().forward(x)
        return self.policy_head(features)


class ValueNetwork(BaseNetwork):
    def __init__(self, state_dim, hidden_layers=[64, 64]):
        super(ValueNetwork, self).__init__(state_dim, hidden_layers)
        self.value_head = nn.Sequential(nn.Linear(self.input_dim, 1))

    def forward(self, x):
        features = super().forward(x)
        return self.value_head(features).squeeze()


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
        self.max_timesteps = 500

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, hidden_layers=hidden_layers
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_layers=hidden_layers)
            self.value_optimizer = optim.Adam(
                self.value_net.parameters(), lr=learning_rate
            )

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def train(self, episodes=1000, gamma=0.99):
        rewards_per_episode = []
        progress_bar = tqdm(range(episodes), desc="Training Progress")

        for episode in progress_bar:
            state = self.env.reset()[0]
            log_probs = []
            rewards = []
            states = []
            total_reward = 0
            done = False

            for t in range(self.max_timesteps):
                action, log_prob = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_state

                if done:
                    break

            returns = self.compute_returns(rewards, gamma)
            states_tensor = torch.FloatTensor(np.array(states))

            if self.use_baseline:
                value_preds = self.value_net(states_tensor)
                value_loss = nn.MSELoss()(value_preds, returns)

                with torch.no_grad():
                    baseline_vals = self.value_net(states_tensor)
                advantages = returns - baseline_vals
            else:
                advantages = returns

            policy_loss = -torch.sum(torch.stack(log_probs) * advantages)
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if self.use_baseline:
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

            rewards_per_episode.append(total_reward)
            progress_bar.set_postfix({"reward": total_reward})

        return rewards_per_episode

    def save_model(self, model_name):
        checkpoint = {
            "state_dict": self.policy_net.state_dict(),
            "hidden_layers": self.policy_net.hidden_layers,
            "use_baseline": self.use_baseline,
        }
        model_path = os.path.join(f"{model_name}.pth")
        torch.save(checkpoint, model_path)
        print(f"MC REINFORCE model saved to {model_path}\n")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_layers=checkpoint["hidden_layers"],
        )
        self.policy_net.load_state_dict(checkpoint["state_dict"])
        self.policy_net.eval()
        self.use_baseline = checkpoint["use_baseline"]
        if self.use_baseline:
            self.value_net = ValueNetwork(
                self.state_dim, hidden_layers=checkpoint["hidden_layers"]
            )
            self.value_net.eval()
