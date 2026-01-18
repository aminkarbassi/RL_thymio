import time
import json
from collections import deque
from tdmclient import ClientAsync, aw

# =========================================================
# CONSTANTS
# =========================================================
FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2

FORWARD_SPEED = 220
TURN_SPEED = 150
DT = 0.1

# =========================================================
# OPENING DETECTION PARAMETERS (RAW SENSOR SPACE)
# =========================================================
HISTORY_LEN = 10
OPEN_STREAK = 4

# Thymio raw sensor values: higher = closer
# These thresholds correspond to the training thresholds converted to raw values
WALL_NEAR_RAW = 2000     # ~0.15m in training
OPEN_FAR_RAW  = 2000     # ~0.25m in training

# =========================================================
# LOAD Q-TABLE
# =========================================================
with open("thymio_q_table.json") as f:
    Q = {eval(k): v for k, v in json.load(f).items()}

print(f"Loaded Q-table with {len(Q)} states")

# =========================================================
# DISCRETIZATION (RAW → SYMBOLIC)
# Matches training: 0=far, 1=medium, 2=close, 3=very close
# =========================================================
def discretize_left_right(v):
    """Discretize side sensors - matches training thresholds [0.05, 0.10, 0.2]"""
    if v == 0 or v < 500:
        return 0   # FAR (> 0.2m)
    elif v < 1500:
        return 1   # MEDIUM (0.1-0.2m)
    elif v < 2000:
        return 2   # CLOSE (0.05-0.1m)
    else:
        return 3   # VERY CLOSE (< 0.05m)

def discretize_center_left_right(v):
    """Discretize center-left/center-right sensors - matches [0.05, 0.15, 0.25]"""
    if v == 0 or v < 600:
        return 0   # FAR (> 0.25m)
    elif v < 1200:
        return 1   # MEDIUM (0.15-0.25m)
    elif v < 2500:
        return 2   # CLOSE (0.05-0.15m)
    else:
        return 3   # VERY CLOSE (< 0.05m)

def discretize_center(v):
    """Discretize center sensor - matches [0.05, 0.2, 0.3]"""
    if v == 0 or v < 500:
        return 0   # FAR (> 0.3m)
    elif v < 900:
        return 1   # MEDIUM (0.2-0.3m)
    elif v < 2000:
        return 2   # CLOSE (0.05-0.2m)
    else:
        return 3   # VERY CLOSE (< 0.05m)

# =========================================================
# OPENING DETECTION (TEMPORAL)
# =========================================================
def detect_opening(side_hist, side):
    """Detect if there's an opening on the specified side."""
    if len(side_hist) < max(OPEN_STREAK + 1, 5):
        return 0

    idx = 0 if side == "left" else 1
    vals = [v[idx] for v in side_hist]

    # For raw Thymio: higher value = closer
    # recently_wall: high values in older readings
    # now_open: low values in recent readings
    recently_wall = any(v > WALL_NEAR_RAW for v in vals[:-OPEN_STREAK])
    now_open = all(v < OPEN_FAR_RAW for v in vals[-OPEN_STREAK:])

    return 1 if (recently_wall and now_open) else 0

# =========================================================
# STATE CONSTRUCTION - MATCHES TRAINING 7-TUPLE
# =========================================================
def get_state(prox, side_hist):
    """
    Create 7-tuple state matching training:
    (left, center_left, center, center_right, right, left_open, right_open)
    """
    # prox.horizontal indices: 0=left, 1=center-left, 2=center, 3=center-right, 4=right
    left = discretize_left_right(prox[0])
    center_left = discretize_center_left_right(prox[1])
    center = discretize_center(prox[2])
    center_right = discretize_center_left_right(prox[3])
    right = discretize_left_right(prox[4])

    left_open = detect_opening(side_hist, "left")
    right_open = detect_opening(side_hist, "right")

    return (left, center_left, center, center_right, right, left_open, right_open)

def choose_action(state, prox):
    """Choose action with safety fallback for unknown states."""
    if state in Q:
        return max(range(3), key=lambda a: Q[state][a])
    else:
        # Safety fallback: avoid walls
        front_max = max(prox[1], prox[2], prox[3])
        if front_max > 2500:  # Close to front wall
            if prox[0] < prox[4]:
                return TURN_LEFT
            else:
                return TURN_RIGHT
        return FORWARD

# =========================================================
# RUN POLICY ON REAL THYMIO
# =========================================================
unknown_states = 0

with ClientAsync() as client:
    node = aw(client.wait_for_node())
    aw(node.lock())

    aw(node.watch(variables={"prox.horizontal"}))

    side_hist = deque(maxlen=HISTORY_LEN)

    print("Running trained policy on real Thymio...")

    try:
        while True:
            aw(client.sleep(DT))
            aw(node.wait_for_variables({"prox.horizontal"}))

            prox = list(node.v.prox.horizontal)
            if prox is None:
                continue

            # Store raw side distances for opening detection
            side_hist.append((prox[0], prox[4]))

            state = get_state(prox, side_hist)
            
            if state not in Q:
                unknown_states += 1
            
            action = choose_action(state, prox)

            # Motor commands
            if action == FORWARD:
                left, right = FORWARD_SPEED, FORWARD_SPEED
            elif action == TURN_LEFT:
                left, right = -TURN_SPEED, TURN_SPEED
            else:
                left, right = TURN_SPEED, -TURN_SPEED

            aw(node.set_variables({
                "motor.left.target": [left],
                "motor.right.target": [right]
            }))

    except KeyboardInterrupt:
        print(f"\nStopped. Encountered {unknown_states} unknown states.")

    finally:
        aw(node.set_variables({
            "motor.left.target": [0],
            "motor.right.target": [0]
        }))
        aw(node.unlock())

print("Policy execution finished.")
