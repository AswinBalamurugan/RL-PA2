SEEDS = [10, 4, 6, 0, 15]
EPISODES = 1000
TUNING_EPISODES = 200

HYPERPARAM_CONFIG = {
    "Acrobot-v1": {
        "dueling_dqn": {
            "hidden_layers": [[64, 64], [64, 32, 16]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[128, 64], [128, 64, 64]],
            "learning_rates": [1e-3, 5e-3],
        },
    },
    "CartPole-v1": {
        "dueling_dqn": {
            "hidden_layers": [[64, 64], [64, 32, 16]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[128, 64], [128, 64, 32]],
            "learning_rates": [1e-3, 5e-3],
        },
    },
}
