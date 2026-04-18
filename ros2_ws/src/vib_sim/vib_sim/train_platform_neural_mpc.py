import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


class PlatformDynamicsModel(nn.Module):
    def __init__(self):
        super(PlatformDynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class PlatformControlPolicy(nn.Module):
    def __init__(self, max_force=80.0):
        super(PlatformControlPolicy, self).__init__()
        self.max_force = max_force
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * self.max_force


def load_data(path):
    data = np.load(path)
    print(f"Loaded data shape: {data.shape}")
    states = data[:, 1:9]
    forces = data[:, 9:13]
    X_state = states[:-1]
    X_action = forces[:-1]
    Y_next = states[1:]
    print(f"Training samples: {X_state.shape[0]}")
    return X_state, X_action, Y_next


def train_dynamics(X_state, X_action, Y_next):
    print("\n=== Training Dynamics Model (Improved) ===")
    X_s = torch.FloatTensor(X_state)
    X_a = torch.FloatTensor(X_action)
    Y = torch.FloatTensor(Y_next)
    dataset = TensorDataset(X_s, X_a, Y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = PlatformDynamicsModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = nn.MSELoss()
    for epoch in range(500):
        total_loss = 0.0
        count = 0
        for s_batch, a_batch, y_batch in loader:
            pred = model(s_batch, a_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg = total_loss / count
            print(f"  Epoch {epoch+1}/500  Loss: {avg:.6f}")
    save_path = os.path.expanduser("~/ros2_ws/platform_dynamics_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Dynamics model saved to {save_path}")
    return model


def train_policy(dynamics_model, X_state):
    print("\n=== Training Control Policy (Improved) ===")
    eq_pos = -1.0
    horizon = 15
    lam = 0.001
    X_s = torch.FloatTensor(X_state)
    dataset = TensorDataset(X_s)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    policy = PlatformControlPolicy(max_force=80.0)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    dynamics_model.eval()
    for param in dynamics_model.parameters():
        param.requires_grad = False
    for epoch in range(800):
        total_cost = 0.0
        count = 0
        for (s_batch,) in loader:
            state = s_batch
            cost = torch.tensor(0.0)
            for step in range(horizon):
                action = policy(state)
                next_state = dynamics_model(state, action)
                p0 = (next_state[:, 0] - eq_pos) * (next_state[:, 0] - eq_pos)
                p2 = (next_state[:, 2] - eq_pos) * (next_state[:, 2] - eq_pos)
                p4 = (next_state[:, 4] - eq_pos) * (next_state[:, 4] - eq_pos)
                p6 = (next_state[:, 6] - eq_pos) * (next_state[:, 6] - eq_pos)
                pos_err = p0 + p2 + p4 + p6
                v1 = next_state[:, 1] * next_state[:, 1]
                v3 = next_state[:, 3] * next_state[:, 3]
                v5 = next_state[:, 5] * next_state[:, 5]
                v7 = next_state[:, 7] * next_state[:, 7]
                vel_err = v1 + v3 + v5 + v7
                vibration_cost = (pos_err + vel_err).mean()
                effort_cost = (action * action).sum(dim=1).mean()
                cost = cost + vibration_cost + lam * effort_cost
                state = next_state
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            total_cost += cost.item()
            count += 1
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg = total_cost / count
            print(f"  Epoch {epoch+1}/800  Cost: {avg:.4f}")
    save_path = os.path.expanduser("~/ros2_ws/platform_control_policy.pth")
    torch.save(policy.state_dict(), save_path)
    print(f"Control policy saved to {save_path}")
    return policy


def main():
    data_path = os.path.expanduser("~/ros2_ws/platform_training_data.npy")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return
    X_state, X_action, Y_next = load_data(data_path)
    dynamics_model = train_dynamics(X_state, X_action, Y_next)
    train_policy(dynamics_model, X_state)
    print("\n=== Training Complete! ===")


if __name__ == "__main__":
    main()
