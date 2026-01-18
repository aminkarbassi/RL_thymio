# Code using Claude Opus 4.5

import sys
import time
import random
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
# HANDLES
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

# =========================================================
# RL PARAMETERS
# =========================================================
alpha = 0.2
gamma = 0.95
epsilon = 0.8  # Start higher for more exploration
epsilon_decay = 0.99  # Slower decay
epsilon_min = 0.05

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
ACTIONS = [FORWARD, TURN_LEFT, TURN_RIGHT]

# =========================================================
# HISTORY / OPENING DETECTION
# =========================================================
HISTORY_LEN = 10
WALL_NEAR = 0.15
OPEN_FAR = 0.25
OPEN_STREAK = 4

# =========================================================
# Q-TABLE
# =========================================================
Q = {}

def get_Q(state):
    if state not in Q:
        Q[state] = [0.0, 0.0, 0.0]
    return Q[state]

# =========================================================
# STATE - Using all 5 sensors
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

    # Check if there was recently a wall and now it's open
    recently_wall = any(v < WALL_NEAR for v in vals[:-OPEN_STREAK])
    now_open = all(v > OPEN_FAR for v in vals[-OPEN_STREAK:])

    return 1 if (recently_wall and now_open) else 0

def get_state(distances, side_hist):
    """
    Create state using all 5 sensors plus opening detection.
    State: (left, center_left, center, center_right, right, left_open, right_open)
    """
    # Discretize all 5 sensors
    # 0 = far, 1 = medium, 2 = close, 3 = very close
    left = discretize(distances[IDX_L], [0.05, 0.10, 0.2])
    center_left = discretize(distances[IDX_CL], [0.05, 0.15, 0.25])
    center = discretize(distances[IDX_C], [0.05, 0.2, 0.3])
    center_right = discretize(distances[IDX_CR], [0.05, 0.15, 0.25])
    right = discretize(distances[IDX_R], [0.05, 0.10, 0.2])
    # Detect openings
    left_open = detect_opening(side_hist, "left")
    right_open = detect_opening(side_hist, "right")

    return (left, center_left, center, center_right, right, left_open, right_open)

# =========================================================
# REWARD - Improved and balanced
# =========================================================
def compute_reward(distances, action, state, prev_state):
    left_d = distances[IDX_L]
    right_d = distances[IDX_R]
    front_d = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    
    left, center_left, center, center_right, right, left_open, right_open = state

    reward = -5.0  # Base penalty to encourage movement

    # === COLLISION PREVENTION (highest priority) ===
    # Note: discretized values are 0=far, 1=medium, 2=close, 3=very close
    # Strong penalty for being very close to front wall (level 3)
    if center >= 3 or center_left >= 3 or center_right >= 3:
        if action == FORWARD:
            reward -= 20.0  # Heavy penalty for moving toward wall
        else:
            reward += 1.0  # Reward for turning away
    
    # Moderate penalty for approaching front wall (level 2)
    elif center >= 2 or center_left >= 2 or center_right >= 2:
        if action == FORWARD:
            reward -= 10.0
        else:
            reward += 1.0

    # === SIDE WALL PENALTIES ===
    # Level 3 = very close to side walls
    if left >= 3 or right >= 3:
        reward -= 15.0  # Too close to side walls

    # === FORWARD MOVEMENT REWARD ===
    if action == FORWARD and front_d > 0.35:
        reward += 15.0  # Reward forward progress when safe

    # === OPENING DETECTION AND TURNING ===
    # Balanced rewards for both directions
    if left_open and action == TURN_LEFT:
        reward += 30.0
    elif right_open and action == TURN_RIGHT:
        reward += 30.0
    
    # Penalty for ignoring openings
    if left_open and action != TURN_LEFT and front_d < 0.3:
        reward -= 10.0
    if right_open and action != TURN_RIGHT and front_d < 0.3:
        reward -= 10.0

    # === TURNING COST (small to avoid bias) ===
    if action != FORWARD:
        reward -= 2.0  # Small cost for turning

    # === DEAD END HANDLING ===
    # If front is blocked and no openings, encourage any turn
    if (center >= 2) and not left_open and not right_open:
        if action != FORWARD:
            reward += 5.0

    # === CENTERING BONUS ===
    # Encourage staying centered in corridor
    if abs(left_d - right_d) < 0.1 and front_d > 0.4:
        reward += 3.0

    return reward

def front_collision(distances, threshold=0.05):
    """Check if front sensors detect imminent collision."""
    front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    return front_min < threshold

def front_danger(distances, threshold=0.12):
    """Check if robot is dangerously close to front wall."""
    front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    return front_min < threshold

# =========================================================
# POLICY - Improved action selection
# =========================================================
def choose_action(state, distances):
    """Choose action with safety override."""
    # Safety override: if very close to front wall, don't go forward
    front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    
    if random.random() < epsilon:
        if front_min < 0.15:
            # When close to wall, only choose turns
            return random.choice([TURN_LEFT, TURN_RIGHT])
        return random.choice(ACTIONS)
    
    q_values = get_Q(state).copy()
    
    # Mask forward action if too close to wall
    if front_min < 0.12:
        q_values[FORWARD] = -float('inf')
    
    return q_values.index(max(q_values))

def update_Q(state, action, reward, next_state):
    q = get_Q(state)
    q_next = get_Q(next_state)
    q[action] += alpha * (reward + gamma * max(q_next) - q[action])

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

#def emergency_backup():
#    """Back up slightly when too close to wall."""
#    sim.setJointTargetVelocity(left_motor, -forward_speed * 0.5)
#    sim.setJointTargetVelocity(right_motor, -forward_speed * 0.5)
#    time.sleep(0.2)
#    stop_robot()

# =========================================================
# RESET
# =========================================================
def reset_episode():
    stop_robot()
    sim.setObjectPosition(robot, -1, [0, 0, 0])
    sim.setObjectOrientation(robot, -1, [0, 0, 0])
    time.sleep(0.5)

# =========================================================
# TRAINING
# =========================================================
EPISODES = 200
MAX_STEPS = 500
MAX_DIST = 2.0

episode_log = []

print("\nStarting episodic Q-learning with improved front collision handling...\n")

for episode in range(EPISODES):
    reset_episode()

    side_hist = deque(maxlen=HISTORY_LEN)
    total_reward = 0
    collision = False
    prev_state = None

    for step in range(MAX_STEPS):
        # Read all sensors
        distances = []
        for h in prox_handles:
            detected, dist, *_ = sim.readProximitySensor(h)
            distances.append(dist if detected else MAX_DIST)

        side_hist.append((distances[IDX_L], distances[IDX_R]))
        state = get_state(distances, side_hist)

        # Check for danger and potentially back up
        #if front_danger(distances) and step > 0:
        #    emergency_backup()
        # Re-read sensors after backup
        #    distances = []
        #    for h in prox_handles:
        #        detected, dist, *_ = sim.readProximitySensor(h)
        #        distances.append(dist if detected else MAX_DIST)
        #    state = get_state(distances, side_hist)

        action = choose_action(state, distances)
        apply_action(action)
        time.sleep(dt)

        # Read next state
        distances_next = []
        for h in prox_handles:
            detected, dist, *_ = sim.readProximitySensor(h)
            distances_next.append(dist if detected else MAX_DIST)

        side_hist.append((distances_next[IDX_L], distances_next[IDX_R]))

        if front_collision(distances_next):
            collision = True
            reward = -100.0
            q = get_Q(state)
            q[action] += alpha * (reward - q[action])
            total_reward += reward
            break

        next_state = get_state(distances_next, side_hist)
        reward = compute_reward(distances_next, action, next_state, state)
        update_Q(state, action, reward, next_state)

        total_reward += reward
        prev_state = state

    avg_Q = sum(max(v) for v in Q.values()) / max(1, len(Q))
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    episode_log.append({
        "episode": episode,
        "steps": step + 1,
        "total_reward": total_reward,
        "collision": collision,
        "epsilon": epsilon,
        "avg_Q": avg_Q,
        "q_table_size": len(Q),
    })

    print(
        f"Episode {episode:03d} | "
        f"Steps {step+1:3d} | "
        f"Reward {total_reward:8.1f} | "
        f"AvgQ {avg_Q:7.2f} | "
        f"Collision {collision} | "
        f"Epsilon {epsilon:.3f} | "
        f"States {len(Q)}"
    )

stop_robot()

with open("thymio_q_table.json", "w") as f:
    json.dump({str(k): v for k, v in Q.items()}, f, indent=2)

with open("training_log.json", "w") as f:
    json.dump(episode_log, f, indent=2)

print("\nTraining finished.")
print("Saved: thymio_q_table.json")
print("Saved: training_log.json")