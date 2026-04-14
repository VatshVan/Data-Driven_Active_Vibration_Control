import torch
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
import os
from model import MIMO_AdversarialPlant, SpatialMPCOrchestrator, execute_mimo_simulation, calculate_spatial_metrics

def execute_spatial_verification():
    # Simulation Hyperparameters
    dt = 0.01
    steps_per_scenario = 400
    N = 20 # MPC Horizon
    
    # Plant Physical Geometry (e.g., a 1.0m x 0.8m rectangular plate)
    L, W = 0.5, 0.4 
    pos_arr = [(L, W), (L, -W), (-L, W), (-L, -W)]
    
    # Nominal Parameters
    m_nom = 12.0
    Ixx_nom = 2.5
    Iyy_nom = 3.2
    k_arr = [150.0, 150.0, 150.0, 150.0]
    c_arr = [20.0, 20.0, 20.0, 20.0]
    
    # MPC Tuning Matrices
    # Penalize deviations in z, phi, theta heavily. Velocities less so.
    Q = sp.csc_matrix(np.diag([100.0, 80.0, 80.0, 5.0, 5.0, 5.0]))
    # Penalize aggressive individual actuator usage to prevent harmonic fighting
    R = sp.csc_matrix(np.diag([0.5, 0.5, 0.5, 0.5]))
    
    u_min = [-50.0, -50.0, -50.0, -50.0]
    u_max = [50.0, 50.0, 50.0, 50.0]

    print("\nInitializing Spatial HyperMPC Orchestrator...")
    orchestrator = SpatialMPCOrchestrator(
        m_init=m_nom, Ixx=Ixx_nom, Iyy=Iyy_nom, 
        k_arr=k_arr, c_arr=c_arr, pos_arr=pos_arr, 
        N=N, Q=Q, R=R, u_min=u_min, u_max=u_max, dt=dt
    )

    print("Initializing Non-Stationary Adversarial Plant...")
    Q_v = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]) # Sensor Noise
    plant = MIMO_AdversarialPlant(
        m_nom=m_nom, Ixx_nom=Ixx_nom, Iyy_nom=Iyy_nom, 
        k_arr=k_arr, c_arr=c_arr, pos_arr=pos_arr, 
        dt=dt, noise_cov=Q_v, param_drift_rate=0.05
    )

    # Define the 6 Verification Targets: [z, phi, theta, z_dot, phi_dot, theta_dot]
    targets = {
        "1. Pure Heave (+Z)":          [0.2,  0.0,   0.0,  0.0, 0.0, 0.0],
        "2. Pure Roll (+Phi)":         [0.0,  0.15,  0.0,  0.0, 0.0, 0.0],
        "3. Pure Pitch (-Theta)":      [0.0,  0.0,  -0.15, 0.0, 0.0, 0.0],
        "4. Coupled Heave & Roll":     [0.15, -0.1,  0.0,  0.0, 0.0, 0.0],
        "5. Coupled Heave & Pitch":    [0.15, 0.0,   0.1,  0.0, 0.0, 0.0],
        "6. Full Spatial Translation": [0.25, 0.1,  -0.1,  0.0, 0.0, 0.0]
    }

    records = []
    trajectories = {}
    
    # Start all scenarios from origin
    x_init = torch.zeros(6, dtype=torch.float32)

    for name, target in targets.items():
        print(f"Executing Scenario: {name}")
        
        # We need to reset the plant's internal state to prevent 
        # extreme drift accumulation between independent tests
        plant.m_true = m_nom
        
        X, U, M_track = execute_mimo_simulation(
            orchestrator, plant, steps=steps_per_scenario, 
            x_init=x_init, target_state=target
        )
        
        itae, tv, max_err, settle, m_rmse = calculate_spatial_metrics(X, U, M_track, dt)
        
        records.append({
            'Scenario': name,
            'ITAE_Total': round(itae, 4),
            'Control_TV': round(tv, 4),
            'Max_Z_Error': round(max_err, 4),
            'Settling_Time': round(settle, 4),
            'Mass_Est_RMSE': round(m_rmse, 4)
        })
        
        trajectories[name] = {
            'X': X, 
            'U': U, 
            'Target': target,
            'M_track': M_track
        }

    df_results = pd.DataFrame(records)
    print("\n================ MIMO ROBUSTNESS VERIFICATION MATRIX ================\n")
    print(df_results.to_markdown(index=False))
    print("\n=====================================================================\n")

    telemetry_payload = {
        'metrics_dataframe': df_results,
        'trajectory_samples': trajectories,
        'dt': dt
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/spatial_telemetry.pkl', 'wb') as f:
        pickle.dump(telemetry_payload, f)
        
    print("MIMO Analytics run complete. Artifacts saved locally.")

if __name__ == "__main__":
    execute_spatial_verification()