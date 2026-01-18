import sys
import time
import json
from collections import deque

# =========================================================
# ZMQ REMOTE API PATH
# =========================================================
sys.path.append(
    "/home/amin/Documents/ZHAW/RL/project/CoppeliaSim_Edu/programming/zmqRemoteApi/clients/python/src"
)
import coppeliasim_zmqremoteapi_client as remote_api

# =========================================================
# CONNECT
# =========================================================
client = remote_api.RemoteAPIClient()
sim = client.getObject("sim")

# =========================================================
# HANDLES (matching training script paths)
# =========================================================
robot = sim.getObject("/Thymio")
left_motor = sim.getObject("/Thymio/LeftMotor")
right_motor = sim.getObject("/Thymio/RightMotor")

sensor_names = [
    "/Thymio/ProximityLeft",
    "/Thymio/ProximityCenterLeft",
    "/Thymio/ProximityCenter",
    "/Thymio/ProximityCenterRight",
    "/Thymio/ProximityRight",
]
prox_handles = [sim.getObject(name) for name in sensor_names]

IDX_L = 0
IDX_CL = 1
IDX_C = 2
IDX_CR = 3
IDX_R = 4

# =========================================================
# MOTION
# =========================================================
forward_speed = 2.0
turn_speed = 0.8
dt = 0.1

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2

# =========================================================
# HISTORY / OPENING DETECTION (same as training)
# =========================================================
HISTORY_LEN = 10
WALL_NEAR = 0.15
OPEN_FAR = 0.25
OPEN_STREAK = 4
MAX_DIST = 2.0

# =========================================================
# LOAD Q-TABLE
# =========================================================
def load_Q_table(filename="thymio_q_table.json"):
    with open(filename, "r") as f:
        raw = json.load(f)

    Q = {}
    for k, v in raw.items():
        Q[eval(k)] = v
    return Q

Q = load_Q_table()
print(f"Loaded Q-table with {len(Q)} states")

# =========================================================
# STATE (matching training script exactly)
# =========================================================
def discretize(d, thresholds):
    """Discretize distance into levels based on thresholds."""
    for i, thresh in enumerate(thresholds):
        if d < thresh:
            return len(thresholds) - i
    return 0

def detect_opening(side_hist, side):
    """Detect if there's an opening on the specified side."""
    if len(side_hist) < max(OPEN_STREAK + 1, 5):
        return 0

    idx = 0 if side == "left" else 1
    vals = [v[idx] for v in side_hist]

    recently_wall = any(v < WALL_NEAR for v in vals[:-OPEN_STREAK])
    now_open = all(v > OPEN_FAR for v in vals[-OPEN_STREAK:])

    return 1 if (recently_wall and now_open) else 0

def get_state(distances, side_hist):
    """
    Create state using all 5 sensors plus opening detection.
    State: (left, center_left, center, center_right, right, left_open, right_open)
    """
    left = discretize(distances[IDX_L], [0.05, 0.10, 0.2])
    center_left = discretize(distances[IDX_CL], [0.05, 0.15, 0.25])
    center = discretize(distances[IDX_C], [0.05, 0.2, 0.3])
    center_right = discretize(distances[IDX_CR], [0.05, 0.15, 0.25])
    right = discretize(distances[IDX_R], [0.05, 0.10, 0.2])

    left_open = detect_opening(side_hist, "left")
    right_open = detect_opening(side_hist, "right")

    return (left, center_left, center, center_right, right, left_open, right_open)

# =========================================================
# ACTIONS
# =========================================================
def apply_action(action):
    if action == FORWARD:
        sim.setJointTargetVelocity(left_motor, forward_speed)
        sim.setJointTargetVelocity(right_motor, forward_speed)
    elif action == TURN_LEFT:
        sim.setJointTargetVelocity(left_motor, -turn_speed)
        sim.setJointTargetVelocity(right_motor, turn_speed)
    elif action == TURN_RIGHT:
        sim.setJointTargetVelocity(left_motor, turn_speed)
        sim.setJointTargetVelocity(right_motor, -turn_speed)

def stop_robot():
    sim.setJointTargetVelocity(left_motor, 0)
    sim.setJointTargetVelocity(right_motor, 0)

# =========================================================
# RUN TRAINED POLICY
# =========================================================
print("\nRunning trained policy (epsilon = 0.0)\n")

side_hist = deque(maxlen=HISTORY_LEN)
unknown_states = 0

try:
    while True:
        distances = []
        for h in prox_handles:
            detected, dist, *_ = sim.readProximitySensor(h)
            distances.append(dist if detected else MAX_DIST)

        side_hist.append((distances[IDX_L], distances[IDX_R]))
        state = get_state(distances, side_hist)

        # Greedy action with safety fallback
        if state in Q:
            action = Q[state].index(max(Q[state]))
        else:
            unknown_states += 1
            # Safety fallback for unknown states
            front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
            if front_min < 0.15:
                # Turn away from closer side
                if distances[IDX_L] > distances[IDX_R]:
                    action = TURN_LEFT
                else:
                    action = TURN_RIGHT
            else:
                action = FORWARD

        apply_action(action)
        time.sleep(dt)

except KeyboardInterrupt:
    print(f"\nStopped. Encountered {unknown_states} unknown states.")
    stop_robot()