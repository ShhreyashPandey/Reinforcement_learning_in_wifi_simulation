import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from gym_env.wifi_capture_env import WifiCaptureEnv

# Create output directory
os.makedirs("models", exist_ok=True)

# Step 1: Create environment
env = WifiCaptureEnv(num_aps=30, max_steps=20, threat_level="medium")

# Step 2: Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 3: Define PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    device=device,           
    n_steps=128,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.55,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.5
)

# Step 4: Train the model
TIMESTEPS = 1000000
print(f"Training PPO for {TIMESTEPS} timesteps on {device.upper()}...")
model.learn(total_timesteps=TIMESTEPS)

# Step 5: Save
model.save("models/ppo_model")
print("PPO model saved to models/ppo_model.zip")
