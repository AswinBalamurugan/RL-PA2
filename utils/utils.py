import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from algorithms.dueling_dqn import DQNAgent
from algorithms.mc_reinforce import MCREINFORCE
import seaborn as sns
import scienceplots
from utils.config import SEEDS

plt.style.use(["ieee", "no-latex"])
sns.set_palette("colorblind")
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.titlesize"] = 20
sns.set_style("dark")


def run_experiment(
    env_name, algorithm, variant, seed, hidden_layers, learning_rate, episodes
):
    env = gym.make(env_name)
    if algorithm == "dueling_dqn":
        agent = DQNAgent(
            env,
            dueling_type=variant,
            seed=seed,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
        )
    elif algorithm == "mc_reinforce":
        agent = MCREINFORCE(
            env,
            use_baseline=(variant == "with_baseline"),
            seed=seed,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
        )
    rewards = agent.train(episodes=episodes)
    env.close()
    return rewards, agent


def plot_results(results_dict, env_name, algorithm, plot_name=None):
    plt.figure(figsize=(12, 6))
    for variant, rewards in results_dict.items():
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        plt.plot(mean_rewards, label=f"{algorithm} {variant}")
        plt.fill_between(
            range(len(mean_rewards)),
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )
        moving_avg = np.convolve(mean_rewards, np.ones(50) / 50, mode="valid")
        plt.plot(
            range(49, len(mean_rewards)),
            moving_avg,
            linestyle="--",
            label=f"{variant} Moving Avg (50 ep)",
        )
    plt.title(f"{algorithm} on {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if plot_name is None:
        plot_name = f"{env_name}_{algorithm}_results"
    plot_path = os.path.join("results", "plots", env_name, f"{plot_name}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def save_rewards_to_csv(rewards_dict, env_name, algorithm):
    os.makedirs(os.path.join("results", "csv"), exist_ok=True)
    for variant, rewards in rewards_dict.items():
        df = pd.DataFrame(rewards.T, columns=[f"Seed {seed}" for seed in SEEDS])
        df.to_csv(
            os.path.join(
                "results", "csv", f"{env_name}_{algorithm}_{variant}_rewards.csv"
            ),
            index=False,
        )
