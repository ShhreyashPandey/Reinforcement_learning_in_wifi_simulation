#  Purpose:
# Randomly creates signal strengths, channel interference, etc.
# A random AP generator to simulate new Wi-Fi environments.
# Used inside wifi_capture_env.py to simulate varied conditions every episode

import random

def generate_access_points(num_aps, threat_level='medium'):
    """
    Generates a list of simulated Wi-Fi APs with realistic traits.

    AP types:
    - Honeypots: fake, decoys (20%)
    - Targets: real with PMKID/handshake (50%)
    - Noise: random background (rest)
    
    Each AP has:
    - SSID
    - Signal [0.3 to 1.0]
    - PMKID available: bool
    - Handshake available: bool
    - Channel [1-11]
    - Honeypot flag
    - Critical target flag
    """
    aps = []
    ap_id = 0

    # Determine ratios based on threat level
    if threat_level == 'high':
        honeypot_ratio = 0.3
        target_ratio = 0.4
    elif threat_level == 'low':
        honeypot_ratio = 0.1
        target_ratio = 0.6
    else:  # medium
        honeypot_ratio = 0.2
        target_ratio = 0.5

    num_honeypots = int(num_aps * honeypot_ratio)
    num_targets = int(num_aps * target_ratio)
    num_background = num_aps - num_honeypots - num_targets

    def make_ap(ssid_prefix, is_honeypot=False, pmkid=False, handshake=False):
        nonlocal ap_id
        ap = {
            "ssid": f"{ssid_prefix}_{ap_id}",
            "signal": round(random.uniform(0.3, 1.0), 2),
            "pmkid_available": pmkid,
            "handshake_available": handshake,
            "channel": random.randint(1, 11),
            "is_honeypot": is_honeypot,
            "critical": random.random() < 0.2  # 20% of APs are marked critical
        }
        ap_id += 1
        return ap

    # Honeypots
    for _ in range(num_honeypots):
        aps.append(make_ap("Honeypot", is_honeypot=True))

    # Real targets
    for _ in range(num_targets):
        pmkid = random.random() < 0.6
        handshake = random.random() < 0.8
        aps.append(make_ap("Target", pmkid=pmkid, handshake=handshake))

    # Noise
    for _ in range(num_background):
        aps.append(make_ap("Noise"))

    random.shuffle(aps)
    return aps
