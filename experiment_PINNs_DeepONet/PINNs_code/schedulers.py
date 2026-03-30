"""
Alpha (PDE loss weight) schedulers with a uniform hook interface.

All schedulers expose the same API so that training.py is completely
type-agnostic:

    scheduler.on_epoch_start(epoch)
    scheduler.on_step(bc_loss_val, pde_loss_val)
    scheduler.on_epoch_end(avg_bc_loss, avg_pde_loss)
    w_bc, w_pde = scheduler.get_weights()

    scheduler.is_per_step    # True only for ReLoBRaLo
    scheduler.in_final_stage # when to start best-model tracking
    scheduler.log_str        # human-readable status for logging
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_alpha_schedule(s: str) -> Dict[int, float]:
    """Parse ``"0:1e-4,30:1e-3,60:1e-2"`` into ``{0: 1e-4, 30: 1e-3, 60: 1e-2}``."""
    schedule: Dict[int, float] = {}
    for item in s.split(','):
        epoch_str, value_str = item.strip().split(':')
        schedule[int(epoch_str)] = float(value_str)
    return schedule


# ---------------------------------------------------------------------------
# FixedAlphaSchedule
# ---------------------------------------------------------------------------

class FixedAlphaSchedule:
    """
    Epoch-based fixed alpha schedule from a ``{epoch: alpha}`` dict.

    ``weight_bc`` is always 1.0; ``weight_pde`` follows the schedule.
    """

    def __init__(self, schedule: Dict[int, float]):
        self.schedule = schedule
        self._alpha = schedule.get(0, 1e-3)
        self._final_epoch = max(schedule.keys()) if len(schedule) > 1 else 0
        self._entered_final = len(schedule) <= 1

    # -- hooks ---------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        if epoch in self.schedule:
            self._alpha = self.schedule[epoch]
        if epoch >= self._final_epoch and not self._entered_final and self._final_epoch > 0:
            self._entered_final = True

    def on_step(self, bc_loss: float, pde_loss: float):
        pass

    def on_epoch_end(self, avg_bc_loss: float, avg_pde_loss: float):
        pass

    def get_weights(self) -> Tuple[float, float]:
        return 1.0, self._alpha

    # -- properties ----------------------------------------------------------

    @property
    def is_per_step(self) -> bool:
        return False

    @property
    def in_final_stage(self) -> bool:
        return self._entered_final

    @property
    def log_str(self) -> str:
        return f"α={self._alpha:.0e}"


# ---------------------------------------------------------------------------
# AlphaPlateauScheduler
# ---------------------------------------------------------------------------

class AlphaPlateauScheduler:
    """
    Automatically increase alpha (PDE loss weight) when BC loss plateaus.

    Monitors the epoch-averaged training BC loss and multiplies alpha by
    ``factor`` when no significant improvement is observed for ``patience``
    consecutive epochs.  After each increase a cooldown period prevents
    premature re-triggering.
    """

    def __init__(
        self,
        alpha_init: float = 1e-5,
        alpha_max: float = 1.0,
        factor: float = 10,
        patience: int = 5,
        cooldown: int = 3,
        min_relative_improvement: float = 0.1,
    ):
        self._alpha = alpha_init
        self.alpha_max = alpha_max
        self.factor = factor
        self.patience = patience
        self.cooldown_epochs = cooldown
        self.min_rel_improv = min_relative_improvement

        self.alpha_init = alpha_init
        self.best_bc = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        self._message: Optional[str] = None

    def reset(self):
        """Reset scheduler to initial state (for iterative multi-stage training)."""
        self._alpha = self.alpha_init
        self.best_bc = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        self._message = None

    # -- hooks ---------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        pass

    def on_step(self, bc_loss: float, pde_loss: float):
        pass

    def on_epoch_end(self, avg_bc_loss: float, avg_pde_loss: float):
        self._message = None

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.best_bc = avg_bc_loss
            return

        if self._alpha >= self.alpha_max:
            return

        threshold = self.best_bc * (1.0 - self.min_rel_improv)
        if avg_bc_loss < threshold:
            self.best_bc = avg_bc_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            old_alpha = self._alpha
            self._alpha = min(self._alpha * self.factor, self.alpha_max)
            self.wait = 0
            self.cooldown_counter = self.cooldown_epochs
            self._message = (
                f"Alpha increased: {old_alpha:.0e} → {self._alpha:.0e} "
                f"(BC={avg_bc_loss:.4e})"
            )

    def get_weights(self) -> Tuple[float, float]:
        return 1.0, self._alpha

    # -- properties ----------------------------------------------------------

    @property
    def is_per_step(self) -> bool:
        return False

    @property
    def in_final_stage(self) -> bool:
        return True

    @property
    def log_str(self) -> str:
        return f"α={self._alpha:.0e}"

    @property
    def message(self) -> Optional[str]:
        """Non-None when alpha just changed -- caller should print this."""
        return self._message


# ---------------------------------------------------------------------------
# ReLoBRaLo
# ---------------------------------------------------------------------------

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback (ReLoBRaLo).

    Dynamically balances BC and PDE loss weights at every training step.

    Reference: Bischof & Kraus (2021), arXiv:2110.09813
    """

    def __init__(
        self,
        num_losses: int = 2,
        alpha: float = 0.999,
        temperature: float = 1.0,
        rho: float = 0.99,
        random_lookback: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.num_losses = num_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.random_lookback = random_lookback
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_history: List[List[float]] = [[] for _ in range(num_losses)]
        self.weights = torch.ones(num_losses, device=self.device) / num_losses
        self.initial_losses: Optional[torch.Tensor] = None
        self._step_count = 0

    # -- hooks ---------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        pass

    def on_step(self, bc_loss: float, pde_loss: float):
        losses = [bc_loss, pde_loss]
        current = torch.tensor(losses, device=self.device)

        if self.initial_losses is None:
            self.initial_losses = current.clone()

        for i, val in enumerate(losses):
            self.loss_history[i].append(val)
        self._step_count += 1

        if self._step_count < 2:
            return

        if (self.random_lookback and random.random() < self.rho
                and self._step_count > 2):
            idx = random.randint(0, self._step_count - 2)
            ref = torch.tensor(
                [self.loss_history[i][idx] for i in range(self.num_losses)],
                device=self.device,
            )
        else:
            ref = self.initial_losses

        eps = 1e-8
        rel = current / (ref + eps)
        log_w = torch.log(rel + eps) / self.temperature
        new_w = torch.softmax(log_w, dim=0)

        self.weights = self.alpha * self.weights + (1 - self.alpha) * new_w
        self.weights = self.weights * self.num_losses / self.weights.sum()

    def on_epoch_end(self, avg_bc_loss: float, avg_pde_loss: float):
        pass

    def get_weights(self) -> Tuple[float, float]:
        return self.weights[0].item(), self.weights[1].item()

    # -- properties ----------------------------------------------------------

    @property
    def is_per_step(self) -> bool:
        return True

    @property
    def in_final_stage(self) -> bool:
        return True

    @property
    def log_str(self) -> str:
        w_bc, w_pde = self.get_weights()
        return f"w_bc={w_bc:.2f} w_pde={w_pde:.2f}"

    def reset(self):
        self.loss_history = [[] for _ in range(self.num_losses)]
        self.weights = torch.ones(self.num_losses, device=self.device) / self.num_losses
        self.initial_losses = None
        self._step_count = 0
