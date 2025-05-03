# A modular reward calculator.
# Purpose:
# Encapsulates reward logic in a clean way
# Lets you modify reward rules in one place (e.g., penalize spam, reward stealth, prioritize signal strength)

import random

def calculate_reward(action, aps, current_channel, deauth_count=0):
    """
    Calculates reward based on the agent's action and Wi-Fi AP context.
    Rewards are:
    - PMKID capture: +2 (Scan), +1 (Deauth)
    - Handshake capture: +1 (Scan), +6 (Deauth)
    - Honeypot interaction: -3 penalty
    - Channel switch: +0.3
    - Idle: -0.3
    - Repeated Deauth: extra penalty
    - No honeypot touched: bonus +0.5
    - Critical AP capture bonus: +1.5
    """
    reward = 0.0
    interacted_with_honeypot = False

    for ap in aps:
        if ap["channel"] != current_channel:
            continue

        is_honeypot = ap.get("is_honeypot", False)
        has_pmkid = ap.get("pmkid_available", False)
        has_handshake = ap.get("handshake_available", False)
        is_critical = ap.get("critical", False)

        if action == 0:  # Scan
            if has_pmkid:
                reward += 2
            if has_handshake:
                reward += 1
            if is_honeypot:
                reward -= 3
                interacted_with_honeypot = True

        elif action == 1:  # Deauth
            if has_pmkid:
                reward += 1
            if has_handshake:
                reward += 3
            if is_honeypot:
                reward -= 3
                interacted_with_honeypot = True

        if is_critical and (has_pmkid or has_handshake):
            reward += 1.5  # Bonus for targeting critical APs

    # Action-specific rewards/penalties
    if action == 0:
        reward += 0.5
        reward += random.uniform(0, 0.5)   # scan is stealthy
    elif action == 1:
        reward -= 2.0  # deauth makes noise
        if deauth_count >= 3:
            reward -= (deauth_count - 2) * 2.0  # spam deauth = detection
    elif action == 2:
        reward += 1.5  # channel hop is strategic
    elif action == 3:
        reward -= 0.3  # idle = wasted opportunity

    # Safe behavior bonus
    if not interacted_with_honeypot:
        reward += 0.5

    reward += random.uniform(0, 0.5) 

    return reward
