import sys
import time
import os
import numpy as np
from collections import deque

# =========================================================
# PYTORCH
# =========================================================
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
NUM_ACTIONS = 3

# =========================================================
# HISTORY / OPENING DETECTION
# =========================================================
HISTORY_LEN = 10
WALL_NEAR = 0.15
OPEN_FAR = 0.25
OPEN_STREAK = 4
MAX_DIST = 2.0

# =========================================================
# DQN PARAMETERS (must match training)
# =========================================================
STATE_DIM = 8  # 5 sensors + 2 openings + 1 last_action
HIDDEN_DIM = 128

# =========================================================
# NEURAL NETWORK (must match training architecture)
# =========================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = "thymio_dqn_model.pt"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found!")
    print("\nYou need to train the model first by running:")
    print("  python RL_DQN_labrinh.py")
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"Files in directory: {[f for f in os.listdir('.') if f.endswith('.pt') or f.endswith('.json')]}")
    sys.exit(1)

# Initialize and load model
model = DQN(STATE_DIM, NUM_ACTIONS, HIDDEN_DIM).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['policy_net'])
model.eval()  # Set to evaluation mode
print(f"Loaded DQN model from {MODEL_PATH}")
print(f"Model epsilon at save time: {checkpoint.get('epsilon', 'N/A')}")

# =========================================================
# STATE FUNCTIONS (must match training)
# =========================================================
def normalize_distance(d):
    """Normalize distance to [0, 1] range."""
    return min(d / MAX_DIST, 1.0)

def detect_opening(side_hist, side):
    """Detect if there's an opening on the specified side."""
    if len(side_hist) < max(OPEN_STREAK + 1, 5):
        return 0.0
    
    idx = 0 if side == "left" else 1
    vals = [v[idx] for v in side_hist]
    
    # Check if there was recently a wall and now it's open
    recently_wall = any(v < WALL_NEAR for v in vals[:-OPEN_STREAK])
    now_open = all(v > OPEN_FAR for v in vals[-OPEN_STREAK:])
    
    return 1.0 if (recently_wall and now_open) else 0.0

def get_state_vector(distances, side_hist, last_action):
    """
    Create continuous state vector for neural network.
    Returns: [left, center_left, center, center_right, right, left_open, right_open, last_action_normalized]
    """
    state = np.array([
        normalize_distance(distances[IDX_L]),
        normalize_distance(distances[IDX_CL]),
        normalize_distance(distances[IDX_C]),
        normalize_distance(distances[IDX_CR]),
        normalize_distance(distances[IDX_R]),
        detect_opening(side_hist, "left"),
        detect_opening(side_hist, "right"),
        last_action / 2.0  # Normalize action to [0, 1]
    ], dtype=np.float32)
    return state

# =========================================================
# ACTION SELECTION (greedy, no exploration)
# =========================================================
def select_action(state, distances):
    """Select best action using trained policy with safety override."""
    front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_tensor).cpu().numpy()[0]
        
        # Safety mask: don't go forward if too close to wall
        if front_min < 0.12:
            q_values[FORWARD] = -float('inf')
        
        action = int(np.argmax(q_values))
        
        return action, q_values

# =========================================================
# MOTOR ACTIONS
# =========================================================
def apply_action(action):
    """Apply motor commands based on action."""
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
    """Stop the robot."""
    sim.setJointTargetVelocity(left_motor, 0)
    sim.setJointTargetVelocity(right_motor, 0)

def get_action_name(action):
    """Get human-readable action name."""
    names = {FORWARD: "FORWARD", TURN_LEFT: "LEFT", TURN_RIGHT: "RIGHT"}
    return names.get(action, "UNKNOWN")

# =========================================================
# RUN TRAINED POLICY
# =========================================================
print("\n" + "="*60)
print("Running Trained DQN Policy")
print("="*60)
print("Press Ctrl+C to stop")
print("="*60 + "\n")

side_hist = deque(maxlen=HISTORY_LEN)
last_action = FORWARD
step_count = 0
total_forward = 0
total_left = 0
total_right = 0

try:
    while True:
        # Read sensors
        distances = []
        for h in prox_handles:
            detected, dist, *_ = sim.readProximitySensor(h)
            distances.append(dist if detected else MAX_DIST)
        
        # Update side history
        side_hist.append((distances[IDX_L], distances[IDX_R]))
        
        # Get current state
        state = get_state_vector(distances, side_hist, last_action)
        
        # Select action using trained policy
        action, q_values = select_action(state, distances)
        
        # Apply action
        apply_action(action)
        
        # Update statistics
        step_count += 1
        if action == FORWARD:
            total_forward += 1
        elif action == TURN_LEFT:
            total_left += 1
        else:
            total_right += 1
        
        # Print status every 10 steps
        if step_count % 10 == 0:
            front_d = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
            print(
                f"Step {step_count:4d} | "
                f"Action: {get_action_name(action):7s} | "
                f"Front: {front_d:.3f} | "
                f"L: {distances[IDX_L]:.3f} | "
                f"R: {distances[IDX_R]:.3f} | "
                f"Q: [{q_values[0]:.2f}, {q_values[1]:.2f}, {q_values[2]:.2f}]"
            )
        
        last_action = action
        time.sleep(dt)

except KeyboardInterrupt:
    print("\n\nStopping robot...")

finally:
    stop_robot()
    
    print("\n" + "="*60)
    print("Deployment Statistics")
    print("="*60)
    print(f"Total steps: {step_count}")
    print(f"Forward actions: {total_forward} ({100*total_forward/max(1,step_count):.1f}%)")
    print(f"Left turns: {total_left} ({100*total_left/max(1,step_count):.1f}%)")
    print(f"Right turns: {total_right} ({100*total_right/max(1,step_count):.1f}%)")
    print("="*60)
