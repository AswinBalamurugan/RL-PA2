import os
from utils.utils import (
    run_experiment,
    plot_results,
    save_rewards_to_csv,
)
from utils.config import SEEDS, EPISODES, TUNING_EPISODES, HYPERPARAM_CONFIG
import numpy as np


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

            avg_reward = np.mean(rewards[-TUNING_EPISODES // 10 :])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_hyperparams = (hidden_layers, learning_rate)

    return best_hyperparams


if __name__ == "__main__":
    os.makedirs(os.path.join("results", "models"), exist_ok=True)
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)
    os.makedirs(os.path.join("results", "csv"), exist_ok=True)

    environments = ["Acrobot-v1"]
    # environments = ["Acrobot-v1", "CartPole-v1"]
    algorithms = {
        "dueling_dqn": [1, 2],
        # "mc_reinforce": ["without_baseline", "with_baseline"],
    }

    for env_name in environments:
        os.makedirs(os.path.join("results", "models", env_name), exist_ok=True)
        os.makedirs(os.path.join("results", "plots", env_name), exist_ok=True)

    for env_name in environments:
        for algorithm, variants in algorithms.items():
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
                rewards_dict[variant] = np.array(all_rewards)
                save_rewards_to_csv(
                    {variant: rewards_dict[variant]}, env_name, algorithm
                )
                best_seed_idx = np.argmax(
                    [np.mean(rewards[-EPISODES // 10 :]) for rewards in all_rewards]
                )
                agents_dict[variant] = all_agents[best_seed_idx]
                best_variant_model_name = os.path.join(
                    "results", "models", env_name, f"{algorithm}_{variant}"
                )
                agents_dict[variant].save_model(best_variant_model_name)
                plot_results(
                    {variant: rewards_dict[variant]},
                    env_name,
                    algorithm,
                    plot_name=f"{algorithm}_{variant}",
                )
            plot_results(
                rewards_dict,
                env_name,
                algorithm,
                plot_name=f"{algorithm}_variants_comparison",
            )
