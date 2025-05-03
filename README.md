# Reinforcement Learning in Wi-Fi Simulation

This project explores the use of **Reinforcement Learning (RL)** to simulate how an AI agent can learn to behave like an ethical hacker or penetration tester in a wireless environment. The goal is to enable the agent to scan Wi-Fi networks, avoid traps (honeypots), collect valuable data (like PMKID and handshakes), and make strategic decisions such as when to switch channels or stop scanning.

---

## Project Overview

In real-world cybersecurity, ethical hackers often analyze Wi-Fi networks to test for vulnerabilities. However, this process is time-consuming, risky (due to honeypots), and requires experience. This project builds an RL-based agent that can learn to mimic such behavior **autonomously**.

We created a custom Gym environment named **`WifiCaptureEnv`** that simulates:

* 30 fake Access Points (APs)
* Each AP has different traits: signal strength, PMKID/handshake availability, honeypot flag, and channel
* The agent can only interact with APs on the current channel

We trained our agent using two RL algorithms:

* **PPO (Proximal Policy Optimization)**
* **Recurrent PPO (PPO + LSTM)**

The Recurrent PPO agent outperformed the standard PPO by remembering past actions and avoiding repeated mistakes.

---

## Basic RL Concepts Used

### State

Each state is a vector representing what the agent observes:

* Signal strength of each AP
* Whether data (PMKID/handshake) is available
* If the AP is a honeypot
* Current Wi-Fi channel

### Actions

The agent can perform one of the following actions at each step:

* `0`: Scan for available data
* `1`: Deauthorize (force handshake)
* `2`: Change channel
* `3`: Idle (do nothing)

### Rewards

| Action   | Outcome                   | Reward |
| -------- | ------------------------- | ------ |
| Scan     | Finds PMKID               | +2     |
|          | Finds Handshake           | +1     |
|          | Hits Honeypot             | -3     |
| Deauth   | Captures Handshake        | +3     |
|          | Finds PMKID               | +1     |
|          | Repeated spam (>=3 times) | -2     |
|          | Hits Honeypot             | -3     |
| Channel  | Strategic switch          | +1.5   |
| Idle     | Wastes a step             | -0.3   |
| General  | Avoid honeypots           | +0.5   |
| Critical | Targeting high-value APs  | +1.5   |

### Episode

An episode consists of a fixed number of steps (e.g., 20). Each step simulates one time unit where the agent takes action, gets a reward, and moves to the next state.

---

## How to Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train_ppo.py  # for standard PPO
python train_lstmppo.py  # for PPO with memory (Recurrent PPO)
```

### 3. Run a Demo (trained model)

```bash
python run_demo.py
```

### 4. Generate Agent Video (optional)

```bash
python agent_video_demo.py  # Run with TRAINED=True or False for before/after comparison
```

---

## Results
* PPO agent: -421.75 average reward (struggled with repeating same actions)
* Recurrent PPO agent: -191.50 average reward (performed better due to
 memory)
* Untrained agent: +5.90 (basically acted randomly)
  
 **Interpretation**:
 PPO learned aggressively but spammed actions
 Recurrent PPO avoided repetition & performed better
 Memory helps in partially observable setting

---

## Applications

* Ethical hacking simulators
* Defense-grade wireless scanning automation
* Cybersecurity training for AI agents
* Testing network defense systems in a safe simulated space

---


Feel free to explore the code and experiment with the reward logic, channel hopping behavior, or add new types of APs. If this sparks interest, let's collaborate on scaling it further for real-world use!
