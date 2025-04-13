import gymnasium as gym
import torch
import os
from algorithms.dueling_dqn import DQNAgent
from algorithms.mc_reinforce import MCREINFORCE
import warnings

warnings.filterwarnings("ignore")

# Constants
RESULTS_DIR = "results"
MODELS_DIR = "models"
MODEL_EXT = ".pth"
DUELING_DQN = "dueling_dqn"
MC_REINFORCE = "mc_reinforce"
WITHOUT_BASELINE = "without_baseline"
WITH_BASELINE = "with_baseline"


def load_dueling_dqn_model(env_name, model_path, dueling_type):
    env = gym.make(env_name, render_mode="human")
    model_info = torch.load(model_path)
    agent = DQNAgent(
        env, dueling_type=dueling_type, hidden_layers=model_info["hidden_layers"]
    )
    agent.q_net.load_state_dict(model_info["state_dict"])
    return agent, env


def load_mc_reinforce_model(env_name, model_path, use_baseline):
    env = gym.make(env_name, render_mode="human")
    model_info = torch.load(model_path)
    agent = MCREINFORCE(
        env, use_baseline=use_baseline, hidden_layers=model_info["hidden_layers"]
    )
    agent.policy_net.load_state_dict(model_info["state_dict"])
    return agent, env


def watch_dueling_dqn_agent(agent, env, episodes=5):
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        t = 0
        while not done and t < 500:
            with torch.no_grad():
                action = (
                    agent.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                )
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            t += 1
        print(f"Dueling DQN Episode {ep + 1} Return: {total_reward}")
    env.close()


def watch_mc_reinforce_agent(agent, env, episodes=5):
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        t = 0
        while not done and t < 500:
            action_probs = agent.policy_net(torch.FloatTensor(state))
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            t += 1
        print(f"MC REINFORCE Episode {ep + 1} Return: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Ask user for environment choice
    env_options = ["Acrobot-v1", "CartPole-v1"]
    print("Select Environment:")
    for i, env in enumerate(env_options, 1):
        print(f"{i}. {env}")
    env_choice = int(input("Enter the number of the environment: ")) - 1
    env_name = env_options[env_choice]

    # Ask user for algorithm choice
    algo_options = [DUELING_DQN, MC_REINFORCE]
    print("\nSelect Algorithm:")
    for i, algo in enumerate(algo_options, 1):
        print(f"{i}. {algo}")
    algo_choice = int(input("Enter the number of the algorithm: ")) - 1
    algorithm = algo_options[algo_choice]

    # Ask user for variant choice
    if algorithm == DUELING_DQN:
        variant_options = [1, 2]
        print("\nSelect Dueling DQN Variant:")
        for i, variant in enumerate(variant_options, 1):
            print(f"{i}. Variant {variant}")
        variant_choice = int(input("Enter the number of the variant: ")) - 1
        dueling_type = variant_options[variant_choice]
        model_path = os.path.join(
            RESULTS_DIR,
            MODELS_DIR,
            env_name,
            f"{DUELING_DQN}_{dueling_type}{MODEL_EXT}",
        )
        agent, env = load_dueling_dqn_model(env_name, model_path, dueling_type)
        watch_dueling_dqn_agent(agent, env)

    elif algorithm == MC_REINFORCE:
        variant_options = [WITHOUT_BASELINE, WITH_BASELINE]
        print("\nSelect MC REINFORCE Variant:")
        for i, variant in enumerate(variant_options, 1):
            print(f"{i}. {variant}")
        variant_choice = int(input("Enter the number of the variant: ")) - 1
        use_baseline = variant_options[variant_choice] == WITH_BASELINE
        model_path = os.path.join(
            RESULTS_DIR,
            MODELS_DIR,
            env_name,
            f"{MC_REINFORCE}_{variant_options[variant_choice]}{MODEL_EXT}",
        )
        agent, env = load_mc_reinforce_model(env_name, model_path, use_baseline)
        watch_mc_reinforce_agent(agent, env)
