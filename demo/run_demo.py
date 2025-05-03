import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gym_env.wifi_capture_env import WifiCaptureEnv

action_map = ['Scan', 'Deauth', 'Change Channel', 'Idle']

def run_episode(agent_name, model, is_recurrent=False):
    print(f"\nRunning {agent_name}...\n")
    env = WifiCaptureEnv(num_aps=30, max_steps=20, threat_level="medium")
    obs, _ = env.reset()
    obs = np.array(obs).reshape(1, -1)
    done = False
    step = 0
    rewards = []

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    if is_recurrent:
        # Check for compatibility
        try:
            lstm_state = model.policy.get_initial_state(batch_size=1)
        except AttributeError:
            try:
                lstm_state = model.policy.initial_state
            except AttributeError:
                print("LSTM state init not available, disabling recurrent mode")
                is_recurrent = False

    while not done:
        step += 1
        if is_recurrent:
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
            episode_start = np.array([done])
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.array(obs).reshape(1, -1)
        done = terminated or truncated
        rewards.append(reward)

        channel = info.get("channel", info.get("current_channel", "N/A"))
        print(f"[{agent_name} - Step {step:02}] Action: {action_map[action[0]]}, Reward: {reward:.2f}, Channel: {channel}")
        time.sleep(0.1)

    total_reward = sum(rewards)
    print(f"\n{agent_name} Total Reward: {total_reward:.2f}")
    return rewards, total_reward


if __name__ == "__main__":
    # PPO (Trained)
    model_ppo = PPO.load("models/ppo_model")
    rewards_ppo, total_ppo = run_episode("PPO (Trained)", model_ppo)

    # Recurrent PPO (Trained)
    try:
        model_recurrent = RecurrentPPO.load("models/ppo_recurrent_model")
        rewards_recurrent, total_recurrent = run_episode("Recurrent PPO (Trained)", model_recurrent, is_recurrent=True)
    except Exception as e:
        print(f"Recurrent PPO failed: {e}")
        rewards_recurrent = []
        total_recurrent = 0

    # Untrained PPO (Baseline)
    dummy_env = WifiCaptureEnv()
    model_untrained = PPO("MlpPolicy", dummy_env, verbose=0)
    rewards_random, total_random = run_episode("Untrained PPO", model_untrained)

    # Plotting
    plt.figure(figsize=(10, 6))
    if rewards_ppo:
        plt.plot(rewards_ppo, label=f"PPO: {total_ppo:.2f}", marker='o')
    if rewards_recurrent:
        plt.plot(rewards_recurrent, label=f"Recurrent PPO: {total_recurrent:.2f}", marker='s')
    if rewards_random:
        plt.plot(rewards_random, label=f"Untrained PPO: {total_random:.2f}", marker='x')

    plt.title("Agent Step Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("demo/agent_step_rewards.png")
    plt.show()
