# The PPO training script using stable-baselines3.

# Purpose:
# Creates the Gym environment
# Trains a PPO agent
# Logs progress and saves trained model

# This script:
# Loads your custom WifiCaptureEnv
# Initializes a PPO agent from stable-baselines3
# Trains the agent
# Logs progress
# Saves the trained model

# Section           | Purpose
# WifiCaptureEnv() | Load your custom simulated Wi-Fi environment
# Monitor(env)      | Track rewards per episode for logging
# make_vec_env()     | Wrap into vectorized env (required by PPO)
# PPO(...)          | Initialize the agent with standard hyperparameters
# learn(...)         | Start training the agent
# model.save(...)   | Save the trained policy for later use



import os
import torch
from sb3_contrib import RecurrentPPO
from gym_env.wifi_capture_env import WifiCaptureEnv

# Create output directory
os.makedirs("models", exist_ok=True)

#Step 1: Create the environment
env = WifiCaptureEnv(num_aps=30, max_steps=20, threat_level="medium")

#Step 2: Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 3: Create Recurrent PPO model with LSTM
model = RecurrentPPO(
    policy="MlpLstmPolicy",   # VERY important: use LSTM version
    env=env,
    verbose=1,
    device=device,            # GPU support
    n_steps=128,
    batch_size=64,
    gamma=0.55,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.5,
    max_grad_norm=0.5,
)

# Step 4: Train the model
TIMESTEPS = 1000000
print(f"Training Recurrent PPO (LSTM) for {TIMESTEPS} timesteps on {device.upper()}...")
model.learn(total_timesteps=TIMESTEPS)
# Step 5: Save model
model.save("models/ppo_recurrent_model")
print("Recurrent PPO model saved to models/ppo_recurrent_model.zip")

