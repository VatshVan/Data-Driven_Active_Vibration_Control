import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
from MPC import RTINeuralMPC
from State_space import HybridSystemDynamics, AlphaConfig

class AdversarialPlant:
    def __init__(self, dynamics_model, dt, noise_cov, param_variance):
        self.sys = dynamics_model
        self.dt = dt
        self.Q_v = noise_cov
        self.var = param_variance
        self.m_true = self.sys.m * (1.0 + np.random.uniform(-self.var, self.var))
        self.c_true = self.sys.c * (1.0 + np.random.uniform(-self.var, self.var))
        self.k_true = self.sys.k * (1.0 + np.random.uniform(-self.var, self.var))
        self.A_true = torch.tensor([[0.0, 1.0], [-self.k_true/self.m_true, -self.c_true/self.m_true]], dtype=torch.float32)
        self.B_true = torch.tensor([[0.0], [1.0/self.m_true]], dtype=torch.float32)
        self.ekf_residual = np.zeros(2)
        self.ekf_alpha = 0.15

    def _true_physics(self, x, u):
        return x @ self.A_true.T + u @ self.B_true.T

    def _rk4_true(self, x, u):
        k1 = self._true_physics(x, u)
        k2 = self._true_physics(x + 0.5 * self.dt * k1, u)
        k3 = self._true_physics(x + 0.5 * self.dt * k2, u)
        k4 = self._true_physics(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def step(self, x, u, add_impulse=False):
        x_true_next = self._rk4_true(x, u)
        if add_impulse:
            x_true_next[0, 1] += 5.0 
        raw_noise = np.random.multivariate_normal([0, 0], self.Q_v)
        self.ekf_residual = (1.0 - self.ekf_alpha) * self.ekf_residual + self.ekf_alpha * raw_noise
        x_estimated = x_true_next + torch.tensor(self.ekf_residual, dtype=torch.float32)
        return x_estimated

def execute_rigorous_simulation(mpc, plant, steps, x_init):
    X = np.zeros((steps, 2))
    U = np.zeros((steps, 1))
    x_curr = x_init.clone()
    u_prev = torch.zeros(1)
    for k in range(steps):
        u_opt = mpc.step(x_curr)
        delta_u = torch.clamp(u_opt - u_prev, -0.5, 0.5)
        u_applied = u_prev + delta_u
        impulse_flag = (k == int(steps * 0.4))
        X[k] = x_curr.detach().numpy()
        U[k] = u_applied.detach().numpy()
        with torch.no_grad():
            x_curr = plant.step(x_curr.unsqueeze(0), u_applied.unsqueeze(0), add_impulse=impulse_flag).squeeze(0)
        u_prev = u_applied
    return X, U

def calculate_metrics(X, U, dt):
    steps = X.shape[0]
    time_vector = np.arange(steps) * dt
    itae = np.sum(time_vector * np.abs(X[:, 0])) * dt
    control_tv = np.sum(np.abs(np.diff(U[:, 0])))
    max_error = np.max(np.abs(X[:, 0]))
    threshold = 0.02 * np.abs(X[0, 0])
    settled_indices = np.where(np.abs(X[:, 0]) > threshold)[0]
    settling_time = (settled_indices[-1] * dt) if len(settled_indices) > 0 else 0.0
    return itae, control_tv, max_error, settling_time

def run_monte_carlo_suite(mpc_generator, plant_generator, scenario_name, num_runs=50):
    records = []
    x_init = torch.tensor([2.0, 0.0], dtype=torch.float32)
    sample_trajectory = None
    for i in range(num_runs):
        plant = plant_generator() 
        mpc = mpc_generator()
        X, U = execute_rigorous_simulation(mpc, plant, steps=350, x_init=x_init)
        itae, tv, max_err, settle = calculate_metrics(X, U, dt=0.01)
        records.append({
            'Scenario': scenario_name,
            'Run': i,
            'ITAE': itae,
            'TV': tv,
            'MaxError': max_err,
            'SettlingTime': settle
        })
        if i == 0:
            sample_trajectory = {'X': X, 'U': U}
    return pd.DataFrame(records), sample_trajectory
