"""
Interior point sampling and train/validation splitting.
"""

from typing import Callable, Optional, Tuple

import torch


class InteriorSampler:
    """
    Re-sample interior collocation points each epoch.

    Supports ``'square'`` and ``'disc'`` domains built-in.  For other
    geometries (e.g. ellipse), pass a *custom_sampler* callable with
    signature ``(generator, n_points, domain_params) -> Tensor(N, 2)``.
    """

    def __init__(
        self,
        domain_type: str,
        domain_params: dict,
        n_points: int,
        custom_sampler: Optional[Callable] = None,
    ):
        self.domain_type = domain_type
        self.domain_params = domain_params
        self.n_points = n_points
        self.epoch = 0
        self._custom_sampler = custom_sampler

    def sample(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Draw a fresh set of interior points.

        Args:
            seed: explicit seed (default: 1234 + epoch counter)

        Returns:
            Tensor of shape ``(n_sampled, 2)``
        """
        if seed is None:
            seed = 1234 + self.epoch
        self.epoch += 1

        generator = torch.Generator()
        generator.manual_seed(seed)

        if self._custom_sampler is not None and self.domain_type not in ('square', 'disc'):
            return self._custom_sampler(generator, self.n_points, self.domain_params)

        if self.domain_type == 'square':
            return self._sample_square(generator)
        elif self.domain_type == 'disc':
            return self._sample_disc(generator)
        else:
            raise ValueError(
                f"Unknown domain type '{self.domain_type}' and no custom_sampler provided."
            )

    def _sample_square(self, generator: torch.Generator) -> torch.Tensor:
        p = self.domain_params
        x = torch.rand(self.n_points, generator=generator) * (p['x_max'] - p['x_min']) + p['x_min']
        y = torch.rand(self.n_points, generator=generator) * (p['y_max'] - p['y_min']) + p['y_min']
        return torch.stack([x, y], dim=1)

    def _sample_disc(self, generator: torch.Generator) -> torch.Tensor:
        radius = self.domain_params['radius']
        n_sample = int(self.n_points * 1.3)
        x = torch.rand(n_sample, generator=generator) * 2 * radius - radius
        y = torch.rand(n_sample, generator=generator) * 2 * radius - radius
        pts = torch.stack([x, y], dim=1)
        mask = pts[:, 0] ** 2 + pts[:, 1] ** 2 < radius ** 2
        pts = pts[mask]
        if pts.shape[0] > self.n_points:
            pts = pts[:self.n_points]
        return pts


def split_data(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    train_ratio: float = 0.8,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Random train/validation split.

    Returns:
        ``(x_train, x_val, y_train_or_None, y_val_or_None)``
    """
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    n_train = int(n * train_ratio)

    perm = torch.randperm(n, generator=generator)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    x_train = x[train_idx]
    x_val = x[val_idx]

    if y is not None:
        return x_train, x_val, y[train_idx], y[val_idx]
    return x_train, x_val, None, None
