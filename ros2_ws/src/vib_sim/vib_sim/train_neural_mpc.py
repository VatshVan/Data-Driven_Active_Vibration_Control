import numpy as np
import os

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found! Install with: pip3 install torch")


class DynamicsModel(nn.Module):
    """Learns: next_state = f(current_state, action)"""
    def __init__(self, state_dim=2, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ControlPolicy(nn.Module):
    """Learns: action = pi(state) to minimize vibration"""
    def __init__(self, state_dim=2, action_dim=1, max_force=100.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.max_force = max_force

    def forward(self, state):
        return self.net(state) * self.max_force


def load_data():
    """Load training data collected from simulation"""
    data_path = os.path.expanduser('~/ros2_ws/training_data.npy')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Run the data_collector_node first!")
        return None
    
    data = np.load(data_path)
    print(f"Loaded {len(data)} samples")
    print(f"Columns: time, est_pos, est_vel, raw_pos, raw_vel, force")
    print(f"Data shape: {data.shape}")
    
    return data


def train_dynamics_model(data):
    """Train the dynamics model to predict next state"""
    print("\n========== Training Dynamics Model ==========")
    
    # Extract states and actions
    # State = [est_pos, est_vel], Action = [force]
    states = data[:-1, 1:3]      # [est_pos, est_vel] at time t
    actions = data[:-1, 5:6]     # [force] at time t
    next_states = data[1:, 1:3]  # [est_pos, est_vel] at time t+1

    # Convert to tensors
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)

    # Create model
    model = DynamicsModel(state_dim=2, action_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    batch_size = 256
    n_epochs = 200

    for epoch in range(n_epochs):
        # Random batch
        idx = np.random.choice(len(states_t), batch_size)
        s_batch = states_t[idx]
        a_batch = actions_t[idx]
        ns_batch = next_states_t[idx]

        # Forward pass
        pred = model(s_batch, a_batch)
        loss = loss_fn(pred, ns_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  Loss: {loss.item():.6f}")

    # Save model
    save_path = os.path.expanduser('~/ros2_ws/dynamics_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Dynamics model saved to {save_path}")

    return model


def train_control_policy(data, dynamics_model):
    """Train control policy using learned dynamics"""
    print("\n========== Training Control Policy ==========")

    states = data[:, 1:3]
    states_t = torch.tensor(states, dtype=torch.float32)

    # Create policy
    policy = ControlPolicy(state_dim=2, action_dim=1, max_force=100.0)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # Freeze dynamics model
    for param in dynamics_model.parameters():
        param.requires_grad = False

    # Training parameters
    batch_size = 256
    n_epochs = 500
    horizon = 10         # MPC prediction horizon
    lam = 0.01           # Control effort penalty

    for epoch in range(n_epochs):
        # Sample random states
        idx = np.random.choice(len(states_t), batch_size)
        state = states_t[idx]

        total_cost = 0.0

        # Unroll dynamics for H steps (MPC-style)
        for h in range(horizon):
            action = policy(state)
            next_state = dynamics_model(state, action)

            # Cost: minimize vibration + control effort
            # Position should be at spring_reference (-1.0)
            pos_error = next_state[:, 0] - (-1.0)  # deviation from rest
            vel_error = next_state[:, 1]            # velocity should be 0

            vibration_cost = (pos_error ** 2 + vel_error ** 2).mean()
            effort_cost = (action ** 2).mean()

            total_cost += vibration_cost + lam * effort_cost

            state = next_state

        # Optimize
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  Cost: {total_cost.item():.4f}")

    # Save policy
    save_path = os.path.expanduser('~/ros2_ws/control_policy.pth')
    torch.save(policy.state_dict(), save_path)
    print(f"Control policy saved to {save_path}")

    return policy


def main():
    if not TORCH_AVAILABLE:
        print("Install PyTorch first: pip3 install torch")
        return

    # Step 1: Load data
    data = load_data()
    if data is None:
        return

    # Step 2: Train dynamics model
    dynamics_model = train_dynamics_model(data)

    # Step 3: Train control policy
    policy = train_control_policy(data, dynamics_model)

    print("\n========== Training Complete! ==========")
    print("Files saved:")
    print(f"  ~/ros2_ws/dynamics_model.pth")
    print(f"  ~/ros2_ws/control_policy.pth")
    print("\nNext step: Run the neural_controller_node")


if __name__ == '__main__':
    main()
