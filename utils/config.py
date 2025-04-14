SEEDS = [24, 42, 123, 456, 2023]
EPISODES = 2000
TUNING_EPISODES = 200

HYPERPARAM_CONFIG = {
    "Acrobot-v1": {
        "dueling_dqn": {
            "hidden_layers": [[32, 32], [64]],
            "learning_rates": [9e-4, 3e-3],
        },
        "mc_reinforce": {
            "without_baseline": {
                "hidden_layers": [[32], [64]],
                "learning_rates": [9e-4, 1e-4],
            },
            "with_baseline": {
                "hidden_layers": [[256], [128]],
                "learning_rates": [9e-4, 3e-3],
            },
        },
    },
    "CartPole-v1": {
        "dueling_dqn": {
            "hidden_layers": [[32, 32], [32]],
            "learning_rates": [9e-4, 3e-3],
        },
        "mc_reinforce": {
            "without_baseline": {
                "hidden_layers": [[64], [32]],
                "learning_rates": [9e-4, 3e-3],
            },
            "with_baseline": {
                "hidden_layers": [[64], [128]],
                "learning_rates": [9e-4, 3e-3],
            },
        },
    },
}
