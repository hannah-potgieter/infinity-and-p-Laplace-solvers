"""
Training and evaluation utilities for DeepONet.
"""

import os

import torch
import torch.optim as optim


def train_deeponet(
    model,
    train_loader,
    n_epochs,
    lr=1e-4,
    device=None,
    test_data=None,
    grad_clip=1.0,
    show_every=1,
    checkpoint_dir=None,
    checkpoint_name=None,
    eta_min=1e-6,
):
    """
    Train a DeepONet model.

    Args:
        model: ``DeepONet`` instance (already on *device*).
        train_loader: ``DataLoader`` yielding ``(X, Y)`` batches.
        n_epochs: number of training epochs.
        lr: learning rate for Adam (default ``1e-4``).
        device: torch device (inferred from *model* when ``None``).
        test_data: optional list of ``(test_x, test_y, label)`` tuples.
            Each tuple is evaluated every epoch and its MSE is recorded
            under the key ``'test_<label>'`` in the returned dict.
        grad_clip: max gradient norm for ``clip_grad_norm_`` (default 1.0).
        show_every: print frequency in epochs.
        checkpoint_dir: directory for saving checkpoints (e.g.
            ``'outputs/checkpoints'``).  When ``None``, no checkpoints
            are written.
        checkpoint_name: base name for checkpoint files, e.g.
            ``'origin_disc'``.  Produces
            ``deeponet_final_<name>.pt``.
        eta_min: minimum learning rate for ``CosineAnnealingLR``
            (default ``1e-6``).

    Returns:
        ``loss_history`` dict with at least ``'train_loss'`` and one
        ``'test_<label>'`` entry per element in *test_data*.
    """
    if device is None:
        device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=eta_min,
    )

    loss_history = {'train_loss': [], 'lr': []}
    if test_data:
        for _, _, label in test_data:
            loss_history[f'test_{label}'] = []

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Scheduler: CosineAnnealingLR  T_max={n_epochs}  eta_min={eta_min}")

    # header
    header = "Epoch --- Train Loss --- LR"
    if test_data:
        header += " --- " + " --- ".join(lbl for _, _, lbl in test_data)
    print(header)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(X_batch, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]['lr']
        loss_history['train_loss'].append(avg_loss)
        loss_history['lr'].append(cur_lr)

        if test_data:
            model.eval()
            with torch.no_grad():
                for test_x, test_y, label in test_data:
                    tl = model.compute_loss(
                        test_x.to(device), test_y.to(device),
                    )
                    loss_history[f'test_{label}'].append(tl.item())

        scheduler.step()

        if epoch % show_every == 0:
            msg = f"{epoch:5d} --- {avg_loss:.6e} --- {cur_lr:.2e}"
            if test_data:
                for _, _, label in test_data:
                    msg += f" --- {loss_history[f'test_{label}'][-1]:.6e}"
            print(msg)

    # final checkpoint
    if checkpoint_dir and checkpoint_name:
        final_test = {}
        if test_data:
            for _, _, label in test_data:
                final_test[label] = loss_history[f'test_{label}'][-1]
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'final_test_mse': final_test,
        }, os.path.join(
            checkpoint_dir,
            f'deeponet_final_{checkpoint_name}.pt',
        ))

    return loss_history


def evaluate_over_p_range(
    model,
    test_x,
    test_y,
    device,
    p_col_idx=2,
    p_range=None,
    p_normalize=500.0,
):
    """
    Sweep *model* over a range of p values and report MSE.

    ``test_x[:, p_col_idx]`` is overwritten with each p value in turn.

    Returns:
        dict ``{'p': [...], 'mse': [...]}``
    """
    if p_range is None:
        p_range = list(range(5, 501, 5))

    results = {'p': [], 'mse': []}

    model.eval()
    with torch.no_grad():
        for p in p_range:
            x_p = test_x.clone()
            x_p[:, p_col_idx] = p / p_normalize
            mse = model.compute_loss(x_p.to(device), test_y.to(device))
            results['p'].append(p)
            results['mse'].append(mse.item())

    return results
