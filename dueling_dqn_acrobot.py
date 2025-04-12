import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from tqdm import tqdm

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 1000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
START_EPSILON = 1.0
SEEDS = [0, 10, 20, 30, 40]
MAX_TIMESTEPS = 500 # Time limit per episode


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, type=1):
        super(DuelingDQN, self).__init__()
        self.type = type
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value(features)
        advantages = self.advantage(features)
        if self.type == 1:
            adv_mean = advantages.mean(dim=1, keepdim=True)
        else:
            adv_mean = advantages.max(dim=1, keepdim=True)[0]
        return values + (advantages - adv_mean)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size=BATCH_SIZE):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, env, dueling_type=1, seed=0):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.dueling_type = dueling_type

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.q_net = DuelingDQN(self.state_dim, self.action_dim, type=dueling_type).to(device)
        self.target_q_net = DuelingDQN(self.state_dim, self.action_dim, type=dueling_type).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()

        self.epsilon = START_EPSILON

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        with torch.no_grad():
            next_q = self.target_q_net(next_states)
            next_q_val = next_q.max(1)[0]
            target = rewards + GAMMA * next_q_val * (1 - dones)

        q_values = self.q_net(states)
        action_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(action_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        rewards_per_episode = []
        progress_bar = tqdm(range(EPISODES), desc="Training Progress")

        for episode in progress_bar:
            state = self.env.reset(seed=None)[0]
            total_reward = 0

            for t in range(MAX_TIMESTEPS):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.q_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()

                next_state, reward, done, _, _ = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                self.update()

                state = next_state
                total_reward += reward
                if done:
                    break

            rewards_per_episode.append(total_reward)
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

            if episode % TARGET_UPDATE == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            
            # Update progress bar description
            progress_bar.set_postfix({
                'episode': episode,
                'reward': total_reward,
                'epsilon': f'{self.epsilon:.3f}'
            })

        return rewards_per_episode


# Run 5 seeds
def run_experiment(dueling_type):
    all_rewards = []
    progress_bar = tqdm(SEEDS, desc=f"Training Type-{dueling_type}")
    for seed in progress_bar:
        env = gym.make("Acrobot-v1")
        agent = DQNAgent(env, dueling_type=dueling_type, seed=seed)
        rewards = agent.train()
        all_rewards.append(rewards)
        env.close()
        progress_bar.set_postfix({'seed': seed})
    return np.array(all_rewards)


# Plot results
def plot_results(results_dict):
    plt.figure(figsize=(12, 6))
    for dueling_type, rewards in results_dict.items():
        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)
        plt.plot(mean_rewards, label=f"Dueling DQN Type-{dueling_type}")
        plt.fill_between(range(len(mean_rewards)),
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2)

        moving_avg = np.convolve(mean_rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(mean_rewards)), moving_avg, linestyle='--',
                 label=f"Type-{dueling_type} Moving Avg (50 ep)")
    plt.title("Dueling DQN Type-1 vs Type-2 on Acrobot-v1")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Render trained agent
def watch_agent(agent, episodes=5, sleep=0.02):
    env = gym.make("Acrobot-v1", render_mode="human")
    agent.epsilon = 0.0  # Greedy
    progress_bar = tqdm(range(episodes), desc="Watching Agent")
    for ep in progress_bar:
        state = env.reset(seed=None)[0]
        total_reward = 0
        for t in range(MAX_TIMESTEPS):
            with torch.no_grad():
                action = agent.q_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(sleep)
            if done:
                break
        progress_bar.set_postfix({'reward': total_reward})
        print(f"Episode {ep + 1} Return: {total_reward}")
    env.close()


# Train both types
results = {}
for dtype in [1, 2]:
    print(f"\nTraining Dueling DQN Type-{dtype} on Acrobot-v1")
    results[dtype] = run_experiment(dtype)

plot_results(results)

# Watch a trained agent (choose Type-1 or Type-2)
env = gym.make("Acrobot-v1")
agent = DQNAgent(env, dueling_type=1, seed=0)  # Change to type=2 for other variant
agent.train()
watch_agent(agent)
