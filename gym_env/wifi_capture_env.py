# This is your custom OpenAI Gym environment.

# Purpose:
# Simulates a world with Wi-Fi networks
# Simulates APs(access points) with signal strength, channel, and data types (PMKID or handshake)
# Agent sees their signals, past rewards, etc.
# Agent can scan, deauth, switch channel, or idle
# Rewards depend on the success of these actions
# Rewards are calculated based on what the agent captures
# Episode ends after a fixed number of steps



# Component     | Description
# reset()       | Starts a fresh episode with new random APs
# step(action)  | Accepts agent’s action and returns new state, reward, done, info
# render()      | Displays AP info in human-readable form for debugging
# observation   | 1D array of [signal_strength, data_flag] for each AP
# actions       | 0=Scan, 1=Deauth, 2=Change Channel, 3=Idle
# reward logic  | PMKID = +2, Handshake = +1, Deauth cost = -0.5, Channel switch = -0.1


# wifi_capture_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from utils.wifi_generator import generate_access_points
from utils.rewards import calculate_reward

class WifiCaptureEnv(gym.Env):
    def __init__(self, num_aps=30, max_steps=20, threat_level="medium"):
        super(WifiCaptureEnv, self).__init__()

        self.num_aps = num_aps
        self.max_steps = max_steps
        self.threat_level = threat_level
        self.current_step = 0
        self.channel = 1  # starting channel
        self.fixed_aps = None  # Will be used in test mode
        self.repeat_action_count = 0
        self.previous_action = None
        self.deauth_count = 0


        # Observation: signal strength + data flag per AP + current channel
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_aps * 4 + 1,), dtype=np.float32)

        # Actions
        self.action_space = spaces.Discrete(4)
        self._reset_networks()

    def _reset_networks(self):
        if self.fixed_aps:
            self.aps = self.fixed_aps
        else:
            self.aps = generate_access_points(self.num_aps, threat_level=self.threat_level)
        self.channel = 1

    def _get_obs(self):
        obs = []
        for ap in self.aps:
            obs.append(ap["signal"])  # signal strength
            obs.append(1.0 if ap["pmkid_available"] else 0.0)  # separate PMKID
            obs.append(1.0 if ap["handshake_available"] else 0.0)
            obs.append(1.0 if ap["is_honeypot"] else 0.0)  # encode honeypot in state

        obs.append(self.channel / 11.0)
        return np.array(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.deauth_count = 0
        self.repeat_action_count = 0
        self.previous_action = None
        self._reset_networks()
        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        self.current_step += 1

        # Apply environment logic
        if action == 2:  # Change channel
            self.channel = random.randint(1, 11)
            
        if action == 1:
            self.deauth_count += 1
        else:
            self.deauth_count = 0  # Reset if not repeating deauth

        # Calculate reward
        reward = calculate_reward(action, self.aps, self.channel, self.deauth_count)

        if self.previous_action == action:
            self.repeat_action_count += 1
        else:
            self.repeat_action_count = 0

        if self.repeat_action_count >= 2:
            reward -= self.repeat_action_count * 0.5  # penalize repeating the same action 3+ times
        
        self.previous_action = action
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"current_channel": self.channel}

        return self._get_obs(), reward, terminated, truncated, info
    
    
    def render(self, mode="human"):
        if mode == "human":
            print(f"\n[Step {self.current_step}] Channel: {self.channel}")
            for ap in self.aps:
                print(f"  {ap['ssid']} | Signal: {ap['signal']:.2f} | Channel: {ap['channel']} | "
                      f"PMKID: {ap['pmkid_available']} | Handshake: {ap['handshake_available']}")
        elif mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(8, 4))
            signals = [ap["signal"] for ap in self.aps]
            labels = [ap["ssid"] for ap in self.aps]

            ax.bar(labels, signals, color='skyblue')
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Signal Strength")
            ax.set_title(f"Step {self.current_step} | Channel: {self.channel}")
            fig.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
