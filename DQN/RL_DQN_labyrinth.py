import sys
import time
import random
import json
import numpy as np
from collections import deque

# =========================================================
# PYTORCH
# =========================================================
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# ZMQ REMOTE API PATH (edit based on your path)
# =========================================================
sys.path.append(
    "./CoppeliaSim_Edu/programming/zmqRemoteApi/clients/python/src"
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
# DQN HYPERPARAMETERS
# =========================================================
STATE_DIM = 8  # 5 sensors + 2 openings + 1 last_action
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TAU = 0.005  # Soft update coefficient

# =========================================================
# NEURAL NETWORK
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
# REPLAY BUFFER
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# =========================================================
# DQN AGENT
# =========================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        
        # Policy and target networks
        self.policy_net = DQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net = DQN(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def select_action(self, state, distances):
        """Select action with epsilon-greedy and safety override."""
        front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
        
        if random.random() < self.epsilon:
            # Random action with safety consideration
            if front_min < 0.15:
                return random.choice([TURN_LEFT, TURN_RIGHT])
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Safety mask: don't go forward if too close to wall
            if front_min < 0.12:
                q_values[FORWARD] = -float('inf')
            
            return int(np.argmax(q_values))
    
    def train_step(self):
        """Perform one training step using Double DQN."""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values using Double DQN
        with torch.no_grad():
            # Use policy net to select best actions
            next_actions = self.policy_net(next_states).argmax(1)
            # Use target net to evaluate those actions
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update_target(self):
        """Soft update target network: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                               self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

# =========================================================
# STATE FUNCTIONS
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
# REWARD FUNCTION
# =========================================================
def compute_reward(distances, action, last_action):
    """
    Compute reward with:
    - Collision avoidance (highest priority)
    - Forward progress encouragement
    - Oscillation penalty
    - Centering bonus
    """
    left_d = distances[IDX_L]
    right_d = distances[IDX_R]
    front_d = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    
    reward = -0.5  # Small step penalty to encourage efficiency
    
    # === COLLISION PREVENTION (highest priority) ===
    if front_d < 0.05:
        reward -= 20.0  # Very close to collision
    elif front_d < 0.12:
        if action == FORWARD:
            reward -= 10.0  # Penalty for going forward when too close
        else:
            reward += 2.0   # Reward for turning away
    elif front_d < 0.2:
        if action == FORWARD:
            reward -= 3.0   # Mild penalty for approaching wall
    
    # === SIDE WALL PENALTIES ===
    if left_d < 0.05 or right_d < 0.05:
        reward -= 10.0  # Too close to side walls
    elif left_d < 0.1 or right_d < 0.1:
        reward -= 3.0   # Getting close to side walls
    
    # === FORWARD PROGRESS REWARD ===
    if action == FORWARD and front_d > 0.35:
        reward += 5.0  # Good forward progress
    elif action == FORWARD and front_d > 0.25:
        reward += 2.0  # Moderate forward progress
    
    # === OSCILLATION PENALTY ===
    # Penalize switching between left and right turns
    if last_action == TURN_LEFT and action == TURN_RIGHT:
        reward -= 5.0
    elif last_action == TURN_RIGHT and action == TURN_LEFT:
        reward -= 5.0
    
    # === CONSISTENCY BONUS ===
    # Reward going straight when path is clear
    if front_d > 0.4 and action == FORWARD:
        reward += 3.0
    
    # === CENTERING BONUS ===
    # Encourage staying centered in corridor
    side_diff = abs(left_d - right_d)
    if side_diff < 0.1 and front_d > 0.4:
        reward += 2.0
    elif side_diff < 0.05:
        reward += 1.0
    
    # === TURNING COST (small) ===
    if action != FORWARD:
        reward -= 0.3  # Small penalty for turning
    
    # === OPENING DETECTION BONUS ===
    # Will be enhanced when openings are detected in state
    
    return reward

def front_collision(distances, threshold=0.05):
    """Check if front sensors detect imminent collision."""
    front_min = min(distances[IDX_CL], distances[IDX_C], distances[IDX_CR])
    return front_min < threshold

# =========================================================
# ACTIONS
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

def reset_episode():
    """Reset robot position for new episode."""
    stop_robot()
    sim.setObjectPosition(robot, -1, [0, 0, 0.0191])
    sim.setObjectOrientation(robot, -1, [0, 0, 0])
    time.sleep(0.5)

# =========================================================
# TRAINING LOOP
# =========================================================
EPISODES = 300
MAX_STEPS = 500

# Initialize agent
agent = DQNAgent(STATE_DIM, NUM_ACTIONS)
episode_log = []

print("\n" + "="*60)
print("Starting DQN Training")
print("="*60)
print(f"Episodes: {EPISODES}")
print(f"Max steps per episode: {MAX_STEPS}")
print(f"State dimension: {STATE_DIM}")
print(f"Actions: {NUM_ACTIONS}")
print(f"Device: {device}")
print("="*60 + "\n")

try:
    for episode in range(EPISODES):
        reset_episode()
        
        side_hist = deque(maxlen=HISTORY_LEN)
        total_reward = 0
        total_loss = 0
        loss_count = 0
        collision = False
        last_action = FORWARD
        
        for step in range(MAX_STEPS):
            # Read sensors
            distances = []
            for h in prox_handles:
                detected, dist, *_ = sim.readProximitySensor(h)
                distances.append(dist if detected else MAX_DIST)
            
            # Update side history
            side_hist.append((distances[IDX_L], distances[IDX_R]))
            
            # Get current state
            state = get_state_vector(distances, side_hist, last_action)
            
            # Select and apply action
            action = agent.select_action(state, distances)
            apply_action(action)
            time.sleep(dt)
            
            # Read next state sensors
            distances_next = []
            for h in prox_handles:
                detected, dist, *_ = sim.readProximitySensor(h)
                distances_next.append(dist if detected else MAX_DIST)
            
            # Update side history for next state
            side_hist.append((distances_next[IDX_L], distances_next[IDX_R]))
            
            # Check for collision
            done = False
            if front_collision(distances_next):
                collision = True
                done = True
                reward = -50.0  # Large penalty for collision
            else:
                reward = compute_reward(distances_next, action, last_action)
            
            # Get next state
            next_state = get_state_vector(distances_next, side_hist, action)
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the network
            loss = agent.train_step()
            if loss > 0:
                total_loss += loss
                loss_count += 1
            
            # Soft update target network
            agent.soft_update_target()
            
            total_reward += reward
            last_action = action
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Calculate average loss
        avg_loss = total_loss / max(1, loss_count)
        
        # Log episode
        episode_data = {
            "episode": episode,
            "steps": step + 1,
            "total_reward": float(total_reward),
            "avg_loss": float(avg_loss),
            "collision": collision,
            "epsilon": float(agent.epsilon),
            "memory_size": len(agent.memory)
        }
        episode_log.append(episode_data)
        
        # Print progress
        print(
            f"Episode {episode:03d} | "
            f"Steps {step+1:3d} | "
            f"Reward {total_reward:8.1f} | "
            f"Loss {avg_loss:.4f} | "
            f"Collision {str(collision):5s} | "
            f"Epsilon {agent.epsilon:.3f} | "
            f"Memory {len(agent.memory):5d}"
        )
        
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save(f"dqn_checkpoint_{episode+1}.pt")
            
            # Also save training log
            with open("dqn_training_log.json", "w") as f:
                json.dump(episode_log, f, indent=2)

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")

finally:
    stop_robot()
    
    # Save final model
    agent.save("thymio_dqn_model.pt")
    
    # Save training log
    with open("dqn_training_log.json", "w") as f:
        json.dump(episode_log, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print("Saved: thymio_dqn_model.pt")
    print("Saved: dqn_training_log.json")
    print(f"Total episodes completed: {len(episode_log)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Final memory size: {len(agent.memory)}")
    print("="*60)
