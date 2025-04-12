import gymnasium as gym
import torch
from algorithms.dueling_dqn import DQNAgent
from algorithms.mc_reinforce import MCREINFORCE


def load_dueling_dqn_model(env_name, model_path, dueling_type):
    env = gym.make(env_name, render_mode="human")
    agent = DQNAgent(env, dueling_type=dueling_type)
    agent.q_net.load_state_dict(torch.load(model_path))
    return agent, env


def load_mc_reinforce_model(env_name, model_path, use_baseline):
    env = gym.make(env_name, render_mode="human")
    agent = MCREINFORCE(env, use_baseline=use_baseline)
    agent.policy_net.load_state_dict(torch.load(model_path))
    return agent, env


def watch_dueling_dqn_agent(agent, env, episodes=5):
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = (
                    agent.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                )
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
        print(f"Dueling DQN Episode {ep + 1} Return: {total_reward}")
    env.close()


def watch_mc_reinforce_agent(agent, env, episodes=5):
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action_probs = agent.policy_net(torch.FloatTensor(state))
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
        print(f"MC REINFORCE Episode {ep + 1} Return: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Example usage for Dueling DQN
    env_name = "Acrobot-v1"
    model_path = "results/models/Acrobot-v1_dueling_dqn_1.pth"  # Change as needed
    agent, env = load_dueling_dqn_model(env_name, model_path, dueling_type=1)
    watch_dueling_dqn_agent(agent, env)

    # Example usage for MC REINFORCE
    # model_path = "results/models/CartPole-v1_mc_reinforce_with_baseline.pth"  # Change as needed
    # agent, env = load_mc_reinforce_model(env_name, model_path, use_baseline=True)
    # watch_mc_reinforce_agent(agent, env)
