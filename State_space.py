import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap
from dataclasses import dataclass
from typing import Tuple

class SpringMassDamperPINN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 2, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

@dataclass
class AlphaConfig:
    alpha_max: float = 0.5
    alpha_init: float = 0.0
    loss_ref: float = 1e-2
    gain: float = 5.0

class AlphaScheduler:
    def __init__(self, cfg: AlphaConfig = AlphaConfig()):
        self.cfg = cfg
        self._alpha = cfg.alpha_init

    @property
    def alpha(self) -> float:
        return self._alpha

    def update(self, val_loss: float) -> float:
        ratio = val_loss / max(self.cfg.loss_ref, 1e-12)
        raw = torch.sigmoid(torch.tensor(self.cfg.gain * (1.0 - ratio))).item()
        new_alpha = self.cfg.alpha_max * raw
        self._alpha = max(self._alpha, new_alpha)
        return self._alpha

    def reset(self):
        self._alpha = self.cfg.alpha_init

class HybridSystemDynamics(nn.Module):
    A_p: torch.Tensor
    B_p: torch.Tensor

    def __init__(self, m: float, c: float, k: float, pinn: SpringMassDamperPINN | None = None, alpha_cfg: AlphaConfig = AlphaConfig(), device: str | torch.device = "cpu"):
        super().__init__()
        self.m = float(m)
        self.c = float(c)
        self.k = float(k)
        
        self.register_buffer(
            "A_p",
            torch.tensor([[0.0, 1.0], [-k/m, -c/m]], dtype=torch.float32)
        )
        self.register_buffer(
            "B_p",
            torch.tensor([[0.0], [1.0/m]], dtype=torch.float32)
        )
        
        self.pinn = pinn if pinn is not None else SpringMassDamperPINN()
        self._scheduler = AlphaScheduler(alpha_cfg)
        self.to(device)

    def _alpha_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self._scheduler.alpha,
            dtype=torch.float32,
            device=self.A_p.device
        )

    def _f_physics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.A_p.T + u @ self.B_p.T

    def residual_target(self, x_dot: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x_dot - self._f_physics(x, u)

    def residual_loss(self, x_dot: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target = self.residual_target(x_dot, x, u)
        prediction = self.pinn(x, u)
        return F.mse_loss(prediction, target)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha_tensor().to(dtype=x.dtype, device=x.device)
        f_p = self._f_physics(x, u)
        f_d = self.pinn(x, u)
        return f_p + alpha * f_d

    def rk4_step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self(x, u)
        k2 = self(x + 0.5 * dt * k1, u)
        k3 = self(x + 0.5 * dt * k2, u)
        k4 = self(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def rk4_jacobians(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        def step_fn(x_i: torch.Tensor, u_i: torch.Tensor) -> torch.Tensor:
            return self.rk4_step(x_i.unsqueeze(0), u_i.unsqueeze(0), dt).squeeze(0)

        A_k = vmap(jacrev(step_fn, argnums=0))(x, u)
        B_k = vmap(jacrev(step_fn, argnums=1))(x, u)
        
        return A_k, B_k

    def physics_jacobians(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.A_p.clone(), self.B_p.clone()

    def get_alpha(self) -> float:
        return self._scheduler.alpha

    def update_alpha(self, val_loss: float) -> float:
        return self._scheduler.update(val_loss)

    def reset_alpha(self):
        self._scheduler.reset()

    def diagnostics(self) -> dict:
        return {
            "alpha": self.get_alpha(),
            "m": self.m,
            "c": self.c,
            "k": self.k,
            "pinn_params": sum(p.numel() for p in self.pinn.parameters())
        }