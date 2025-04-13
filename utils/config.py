SEEDS = [0, 10, 20, 30, 40]
EPISODES = 1000
TUNING_EPISODES = 200

HYPERPARAM_CONFIG = {
    "Acrobot-v1": {
        "dueling_dqn": {
            "hidden_layers": [[32, 32], [64]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[64, 32], [128]],
            "learning_rates": [1e-3, 5e-3],
        },
    },
    "CartPole-v1": {
        "dueling_dqn": {
            "hidden_layers": [[32, 32], [64]],
            "learning_rates": [1e-4, 1e-3],
        },
        "mc_reinforce": {
            "hidden_layers": [[32, 16], [64]],
            "learning_rates": [1e-3, 5e-3],
        },
    },
}
