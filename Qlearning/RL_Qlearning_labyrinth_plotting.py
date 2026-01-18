import json
import matplotlib.pyplot as plt

# =====================================================
# LOAD LOG
# =====================================================
with open("./Qlearning/training_log.json", "r") as f:
    log = json.load(f)

episodes   = [e["episode"] for e in log]
rewards    = [e["total_reward"] for e in log]
steps      = [e["steps"] for e in log]
collisions = [e["collision"] for e in log]
epsilons   = [e["epsilon"] for e in log]
avg_Qs     = [e["avg_Q"] for e in log]

# =====================================================
# 1) TOTAL REWARD vs EPISODE
# =====================================================
plt.figure()
plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Learning curve: Total reward vs Episode")
plt.grid(True)

# =====================================================
# 2) STEPS PER EPISODE
# =====================================================
plt.figure()
plt.plot(episodes, steps)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per episode")
plt.grid(True)

# =====================================================
# 3) COLLISION RATE (MOVING AVERAGE)
# =====================================================
window = 5
collision_rate = [
    sum(collisions[max(0, i - window):i + 1]) /
    (i - max(0, i - window) + 1)
    for i in range(len(collisions))
]

plt.figure()
plt.plot(episodes, collision_rate)
plt.xlabel("Episode")
plt.ylabel("Collision rate")
plt.title("Collision rate (moving average)")
plt.grid(True)

# =====================================================
# 4) EPSILON DECAY
# =====================================================
plt.figure()
plt.plot(episodes, epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration decay")
plt.grid(True)

# =====================================================
# 5) AVG-Q VALUE GROWTH
# =====================================================
plt.figure()
plt.plot(episodes, avg_Qs)
plt.xlabel("Episode")
plt.ylabel("Average max Q-value")
plt.title("Q-value confidence growth (avg_Q)")
plt.grid(True)

plt.show()
