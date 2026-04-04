import matplotlib.pyplot as plt
import numpy as np

def generate_temporal_dashboard(X, U, A, dt):
    time_vector = np.arange(X.shape[0]) * dt
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(time_vector, X[:, 0], color='blue')
    axs[0].set_ylabel('Displacement')
    axs[0].grid(True)
    
    axs[1].plot(time_vector, X[:, 1], color='orange')
    axs[1].set_ylabel('Velocity')
    axs[1].grid(True)
    
    axs[2].plot(time_vector, U[:, 0], color='green')
    axs[2].set_ylabel('Control Force')
    axs[2].grid(True)
    
    axs[3].plot(time_vector, A, color='red')
    axs[3].set_ylabel('Confidence Alpha')
    axs[3].set_xlabel('Time')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_phase_portrait(X):
    plt.figure(figsize=(8, 6))
    
    plt.plot(X[:, 0], X[:, 1], color='purple')
    plt.scatter(X[0, 0], X[0, 1], color='green', marker='o', s=100)
    plt.scatter(X[-1, 0], X[-1, 1], color='red', marker='x', s=100)
    
    plt.xlabel('Displacement')
    plt.ylabel('Velocity')
    plt.grid(True)
    plt.show()

def generate_actuator_correlation(U, A):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(A, U[:, 0], alpha=0.6, color='teal')
    
    plt.xlabel('Neural Network Confidence Alpha')
    plt.ylabel('Control Force')
    plt.grid(True)
    plt.show()