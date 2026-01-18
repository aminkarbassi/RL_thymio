This project investigates the use of reinforcement learning to enable a Thymio mobile robot to autonomously perform wall-following and side-opening detection using only onboard proximity sensors. The navigation task is formulated without global localization, maps, or handcrafted motion rules, making the learned policies suitable for deployment on a real robot with limited sensing.

Two reinforcement learning approaches are implemented and compared:

Tabular Q-learning, using a discretized state representation derived from proximity sensor distances and short-term memory for opening detection

Deep Q-Networks (DQN), using continuous sensor inputs and neural network function approximation for improved generalization

Training is performed entirely in CoppeliaSim, using the ZMQ Remote API for closed-loop interaction between Python-based learning algorithms and the simulated Thymio robot. The environment consists of walls and openings, requiring the robot to follow boundaries, avoid frontal and lateral collisions, and detect and enter side openings when they appear.

The repository is structured such that Q-learning and DQN are implemented in separate folders, each containing self-contained training scripts and logging utilities. Training performance is evaluated using logged metrics such as cumulative reward, collision rate, episode length, and average Q-values, as well as qualitative behavior analysis through simulation videos.

Learned policies are deployed in a policy-only execution mode, without exploration or learning, and are prepared for transfer to a physical Thymio robot. This project therefore demonstrates a complete reinforcement learning pipeline for embodied robotics, from problem formulation and reward design to training, evaluation, and sim-to-real deployment.
