import torch
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from State_space import HybridSystemDynamics, AlphaConfig
from MPC import RTINeuralMPC
from model import AdversarialPlant, run_monte_carlo_suite

def execute_verification_matrix():
    dt = 0.01
    N = 20
    Q = sp.csc_matrix(np.diag([10.0, 1.0]))
    R = sp.csc_matrix(np.array([[0.1]]))
    u_min = -1.50
    u_max = 1.50

    def mpc_generator():
        alpha_cfg = AlphaConfig(alpha_max=0.5, alpha_init=0.0, loss_ref=1e-2, gain=5.0)
        dynamics = HybridSystemDynamics(m=1.0, c=0.5, k=2.0, alpha_cfg=alpha_cfg)
        return RTINeuralMPC(dynamics, N, Q, R, u_min, u_max, dt)

    def plant_scenario_a():
        alpha_cfg = AlphaConfig(alpha_max=0.5, alpha_init=0.0, loss_ref=1e-2, gain=5.0)
        dynamics = HybridSystemDynamics(m=1.0, c=0.5, k=2.0, alpha_cfg=alpha_cfg)
        return AdversarialPlant(dynamics, dt, noise_cov=np.diag([0.0, 0.0]), param_variance=0.0)

    def plant_scenario_b():
        alpha_cfg = AlphaConfig(alpha_max=0.5, alpha_init=0.0, loss_ref=1e-2, gain=5.0)
        dynamics = HybridSystemDynamics(m=1.0, c=0.5, k=2.0, alpha_cfg=alpha_cfg)
        return AdversarialPlant(dynamics, dt, noise_cov=np.diag([0.0, 0.0]), param_variance=0.25)

    def plant_scenario_c():
        alpha_cfg = AlphaConfig(alpha_max=0.5, alpha_init=0.0, loss_ref=1e-2, gain=5.0)
        dynamics = HybridSystemDynamics(m=1.0, c=0.5, k=2.0, alpha_cfg=alpha_cfg)
        return AdversarialPlant(dynamics, dt, noise_cov=np.diag([1e-5, 1e-4]), param_variance=0.10)

    df_a, traj_a = run_monte_carlo_suite(mpc_generator, plant_scenario_a, "Nominal", 10)
    df_b, traj_b = run_monte_carlo_suite(mpc_generator, plant_scenario_b, "Parametric Mismatch", 10)
    df_c, traj_c = run_monte_carlo_suite(mpc_generator, plant_scenario_c, "Sensor Degradation", 10)

    df_consolidated = pd.concat([df_a, df_b, df_c], ignore_index=True)
    trajectories = {
        "Nominal": traj_a,
        "Parametric Mismatch": traj_b,
        "Sensor Degradation": traj_c
    }
    
    telemetry_payload = {
        'metrics_dataframe': df_consolidated,
        'trajectory_samples': trajectories
    }
    
    with open('verification_telemetry.pkl', 'wb') as f:
        pickle.dump(telemetry_payload, f)
        
    print("Data serialization complete. Payload saved to verification_telemetry.pkl")

if __name__ == "__main__":
    execute_verification_matrix()