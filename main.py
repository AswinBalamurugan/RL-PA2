import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from algorithms.dueling_dqn import DQNAgent
from algorithms.mc_reinforce import MCREINFORCE

# Hyperparameters
SEEDS = [0, 4, 6, 10, 15]
# SEEDS = [42, 123, 456, 789, 1024]
EPISODES = 1000
TUNING_EPISODES = 200
GAMMA = 0.99
# LEARNING_RATES = [1e-4, 1e-3]
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE = 10
# HIDDEN_LAYER_CONFIGS = [[64, 64], [128, 128], [64, 64, 64]]

HYPERPARAM_CONFIG = {
    "Acrobot-v1": {
        "dueling_dqn": {
            "hidden_layers": [[64, 64], [128, 128], [64, 64, 64]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[0], [16, 16]],
            "learning_rates": [1e-5, 1e-4],
        },
    },
    "CartPole-v1": {
        "dueling_dqn": {
            "hidden_layers": [[64, 64], [128, 128], [64, 64, 64]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[0], [16, 16]],
            "learning_rates": [1e-5, 1e-4],
        },
    },
}


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


def find_best_model(rewards_dict, agents_dict):
    best_model_info = None
    best_avg_reward = -np.inf

    for variant, rewards in rewards_dict.items():
        avg_reward = np.mean(rewards[:, -1])
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_info = (variant, agents_dict[variant])

    return best_model_info


def hyperparameter_tuning(env_name, algorithm, variant):
    best_hyperparams = None
    best_avg_reward = -np.inf
    seed = SEEDS[0]
    config = HYPERPARAM_CONFIG[env_name][algorithm]
    total_runs = len(config["hidden_layers"]) * len(config["learning_rates"])
    current_run = 0

    for hidden_layers in config["hidden_layers"]:
        for learning_rate in config["learning_rates"]:
            current_run += 1
            print(
                f"Tuning {algorithm} {variant} on {env_name} - Run {current_run}/{total_runs}"
            )
            rewards, _ = run_experiment(
                env_name,
                algorithm,
                variant,
                seed,
                hidden_layers,
                learning_rate,
                TUNING_EPISODES,
            )

            avg_reward = np.mean(rewards[-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_hyperparams = (hidden_layers, learning_rate)

    return best_hyperparams


if __name__ == "__main__":
    os.makedirs(os.path.join("results", "models"), exist_ok=True)
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    os.makedirs(os.path.join("results", "csv"), exist_ok=True)

    environments = ["Acrobot-v1", "CartPole-v1"]
    algorithms = {
        # "dueling_dqn": [1, 2],
        "mc_reinforce": ["without_baseline", "with_baseline"],
    }

    for env_name in environments:
        os.makedirs(os.path.join("results", "models", env_name), exist_ok=True)
        os.makedirs(os.path.join("results", "plots", env_name), exist_ok=True)

    for env_name in environments:
        for algorithm, variants in algorithms.items():

            # #! To remove: temp code for training necessary models
            # if env_name == "Acrobot-v1" and algorithm == "dueling_dqn":
            #     continue
            rewards_dict = {}
            agents_dict = {}

            for variant in variants:
                best_hidden_layers, best_learning_rate = hyperparameter_tuning(
                    env_name, algorithm, variant
                )
                print(
                    f"\n Best hyperparameters for {algorithm} {variant} on {env_name}: \
                    Hidden Layers: {best_hidden_layers},  \
                    Learning Rate: {best_learning_rate}\n"
                )

                all_rewards = []
                all_agents = []

                for seed in SEEDS:
                    rewards, agent = run_experiment(
                        env_name,
                        algorithm,
                        variant,
                        seed,
                        best_hidden_layers,
                        best_learning_rate,
                        EPISODES,
                    )
                    all_rewards.append(rewards)
                    all_agents.append(agent)

                # Store rewards for all seeds
                rewards_dict[variant] = np.array(all_rewards)
                save_rewards_to_csv(
                    {variant: rewards_dict[variant]}, env_name, algorithm
                )

                # Find the best model among all seeds for this variant
                best_seed_idx = np.argmax(
                    [np.mean(rewards[-10:]) for rewards in all_rewards]
                )
                agents_dict[variant] = all_agents[best_seed_idx]

                # Save the best model for this variant in the environment-specific directory
                best_variant_model_name = os.path.join(
                    "results", "models", env_name, f"{algorithm}_{variant}"
                )
                agents_dict[variant].save_model(best_variant_model_name)

                # Plot individual results for this variant
                plot_results(
                    {variant: rewards_dict[variant]},
                    env_name,
                    algorithm,
                    plot_name=f"{algorithm}_{variant}",
                )

            # Plot comparison results for all variants of this algorithm and environment
            plot_results(
                rewards_dict,
                env_name,
                algorithm,
                plot_name=f"{algorithm}_variants_comparison",
            )
