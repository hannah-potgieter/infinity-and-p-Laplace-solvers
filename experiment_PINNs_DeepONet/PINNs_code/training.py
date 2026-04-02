"""
Unified PINN training loop.

``train_pinn`` is scheduler-agnostic: it only calls the hook methods
defined by all scheduler classes in ``schedulers.py``.

Model contract:
    - ``model.loss_boundary(x_bc, y_bc) -> scalar``
    - ``model.loss_pde(x_pde) -> scalar``
    - (optional) ``model.loss_interface() -> scalar``
    - (optional) ``model.outlier_percentile`` attribute
"""

import copy
import os
import shutil
import time
from typing import List, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader

from .schedulers import FixedAlphaSchedule, AlphaPlateauScheduler
from .plotting import plot_pde_loss_distribution
from .utils import get_device


def train_pinn(
    model,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    x_interior: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    n_epochs: int = 10,
    batch_size_bc: int = 400,
    batch_size_pde: int = 1000,
    lr: float = 1e-3,
    alpha_scheduler=None,
    x_bc_val: Optional[torch.Tensor] = None,
    y_bc_val: Optional[torch.Tensor] = None,
    x_interior_val: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    interior_sampler=None,
    bc_resample_fn=None,
    scheduler_type: Optional[str] = None,
    scheduler_end_factor: float = 0.01,
    plot_pde_every: int = 0,
    example_name: str = '',
    output_dirs: Optional[dict] = None,
    outlier_off_epoch: Optional[int] = None,
    interface_every: int = 5,
    bc_loss_threshold: Optional[float] = None,
    pde_loss_threshold: Optional[float] = None,
    seed: int = 1234,
    device: Optional[torch.device] = None,
):
    """
    Train a PINN model.

    Args:
        model: PINN with ``loss_boundary`` / ``loss_pde`` methods.
        x_bc, y_bc: training boundary points and values.
        x_interior: training interior/collocation points (ignored when
            *interior_sampler* is given).
        x_test, y_test: test data with exact solution.
        n_epochs: number of full passes through the data.
        batch_size_bc: boundary points per optimisation step.
        batch_size_pde: PDE/collocation points per optimisation step.
        lr: learning rate for Adam.
        alpha_scheduler: any scheduler from ``schedulers.py``.  If *None*,
            a default ``FixedAlphaSchedule`` is created.
        x_bc_val, y_bc_val, x_interior_val: validation data.
        patience: early-stopping patience (None = disabled).
        interior_sampler: ``InteriorSampler`` for re-sampling each epoch.
        bc_resample_fn: callable ``() -> (x_bc, y_bc)`` for re-sampling BC
            data each epoch. When given, *x_bc* / *y_bc* are used only for
            the first epoch.
        scheduler_type: LR scheduler ('linear', 'cosine', 'step',
            'exponential', or None).
        scheduler_end_factor: LR end factor for schedulers.
        plot_pde_every: plot PDE loss diagnostics every N epochs (0 = off).
        example_name: label for logging and file names.
        output_dirs: dict with 'plots' / 'checkpoints' paths.
        outlier_off_epoch: disable outlier removal after this epoch (1-indexed).
        interface_every: compute interface loss every N steps.
        bc_loss_threshold: stop when BC loss is below this value.
        pde_loss_threshold: stop when PDE loss is below this value.
        seed: random seed for DataLoader generators.
        device: torch device (auto-detected if None).

    Returns:
        ``(history_dict, training_time_seconds)``
    """
    if device is None:
        device = get_device()

    # -- default alpha scheduler ---------------------------------------------
    if alpha_scheduler is None:
        alpha_scheduler = AlphaPlateauScheduler()

    # -- validation / resampling flags ---------------------------------------
    use_validation = (x_bc_val is not None and y_bc_val is not None
                      and x_interior_val is not None)
    resample_interior = interior_sampler is not None

    # -- optimizer + LR scheduler --------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = None
    if scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=scheduler_end_factor,
            total_iters=n_epochs)
    elif scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * scheduler_end_factor)
    elif scheduler_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, n_epochs // 3), gamma=0.1)
    elif scheduler_type == 'exponential':
        gamma = scheduler_end_factor ** (1.0 / max(n_epochs, 1))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # -- move data to device -------------------------------------------------
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    if use_validation:
        x_bc_val = x_bc_val.to(device)
        y_bc_val = y_bc_val.to(device)
        x_interior_val = x_interior_val.to(device)

    # -- data loaders --------------------------------------------------------
    bc_dataset = TensorDataset(x_bc, y_bc)
    bc_loader = DataLoader(bc_dataset, batch_size=batch_size_bc, shuffle=True,
                           drop_last=False,
                           generator=torch.Generator().manual_seed(seed))

    if resample_interior:
        n_interior_points = interior_sampler.n_points
        interior_loader = None  # created each epoch
    else:
        interior_dataset = TensorDataset(x_interior)
        interior_loader = DataLoader(
            interior_dataset, batch_size=batch_size_pde, shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed))
        n_interior_points = len(x_interior)

    # -- best-model tracking -------------------------------------------------
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    # -- history -------------------------------------------------------------
    history = {
        'total_loss': [], 'bc_loss': [], 'pde_loss': [], 'interface_loss': [],
        'val_loss': [], 'val_bc_loss': [], 'val_pde_loss': [],
        'test_loss': [], 'step': [],
        'weight_bc': [], 'weight_pde': [], 'lr': [], 'alpha': [],
    }

    # -- layout info ---------------------------------------------------------
    n_bc_batches = len(bc_loader)
    n_pde_batches = (n_interior_points + batch_size_pde - 1) // batch_size_pde
    steps_per_epoch = max(n_bc_batches, n_pde_batches)
    total_steps = n_epochs * steps_per_epoch

    has_interface = hasattr(model, 'loss_interface')

    print(f"\nTraining configuration:")
    print(f"  Boundary points (train): {len(x_bc):,} -> {n_bc_batches} batches of {batch_size_bc}")
    print(f"  Interior points (train): {n_interior_points:,} -> {n_pde_batches} batches of {batch_size_pde}")
    if resample_interior:
        print(f"  Interior sampling: RE-SAMPLE each epoch")
    if use_validation:
        print(f"  Boundary points (val): {len(x_bc_val):,}")
        print(f"  Validation loss: BC only")
        if patience:
            print(f"  Early stopping patience: {patience}")
    else:
        print(f"  Validation: Disabled (using final model)")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Loss balancing: {alpha_scheduler.__class__.__name__}")
    if lr_scheduler is not None:
        print(f"  LR scheduler: {scheduler_type} (LR: {lr:.1e} -> {lr * scheduler_end_factor:.1e})")
    else:
        print(f"  LR scheduler: None (constant LR={lr:.1e})")
    print()

    # -- training loop -------------------------------------------------------
    start_time = time.time()
    global_step = 0
    x_interior_epoch = x_interior  # for PDE-loss plotting

    for epoch in range(n_epochs):
        alpha_scheduler.on_epoch_start(epoch)

        # Outlier curriculum
        if (outlier_off_epoch is not None
                and epoch + 1 > outlier_off_epoch
                and getattr(model, 'outlier_percentile', 0) > 0):
            old_pct = model.outlier_percentile
            model.outlier_percentile = 0.0
            print(f"\n>>> Epoch {epoch + 1}: Outlier removal disabled (was {old_pct}%) <<<\n")

        # Re-sample interior points
        if resample_interior:
            x_interior_epoch = interior_sampler.sample()
            interior_dataset = TensorDataset(x_interior_epoch)
            interior_loader = DataLoader(
                interior_dataset, batch_size=batch_size_pde, shuffle=True,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed + epoch))

        # Re-sample BC points
        if bc_resample_fn is not None:
            x_bc_new, y_bc_new = bc_resample_fn()
            bc_dataset = TensorDataset(x_bc_new.to(device), y_bc_new.to(device))
            bc_loader = DataLoader(
                bc_dataset, batch_size=batch_size_bc, shuffle=True,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed + epoch))

        bc_iter = iter(bc_loader)
        pde_iter = iter(interior_loader)

        epoch_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_intf_loss = 0.0
        epoch_weight_bc = 0.0
        epoch_weight_pde = 0.0
        n_steps_this_epoch = 0

        for step in range(steps_per_epoch):
            try:
                x_bc_batch, y_bc_batch = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                x_bc_batch, y_bc_batch = next(bc_iter)

            try:
                (x_pde_batch,) = next(pde_iter)
            except StopIteration:
                pde_iter = iter(interior_loader)
                (x_pde_batch,) = next(pde_iter)

            x_bc_batch = x_bc_batch.to(device)
            y_bc_batch = y_bc_batch.to(device)
            x_pde_batch = x_pde_batch.to(device)

            optimizer.zero_grad()

            loss_bc = model.loss_boundary(x_bc_batch, y_bc_batch)
            loss_pde = model.loss_pde(x_pde_batch)

            # Let the alpha scheduler see raw losses
            alpha_scheduler.on_step(loss_bc.item(), loss_pde.item())
            w_bc, w_pde = alpha_scheduler.get_weights()
            loss = w_bc * loss_bc + w_pde * loss_pde

            # Interface loss (domain decomposition)
            loss_intf_val = 0.0
            if has_interface and (global_step % interface_every == 0):
                loss_intf = model.loss_interface()
                loss = loss + loss_intf
                loss_intf_val = loss_intf.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bc_loss += loss_bc.item()
            epoch_pde_loss += loss_pde.item()
            epoch_intf_loss += loss_intf_val
            epoch_weight_bc += w_bc
            epoch_weight_pde += w_pde
            n_steps_this_epoch += 1
            global_step += 1

        # -- epoch averages --------------------------------------------------
        avg_loss = epoch_loss / n_steps_this_epoch
        avg_bc_loss = epoch_bc_loss / n_steps_this_epoch
        avg_pde_loss = epoch_pde_loss / n_steps_this_epoch
        avg_intf_loss = epoch_intf_loss / n_steps_this_epoch
        avg_weight_bc = epoch_weight_bc / n_steps_this_epoch
        avg_weight_pde = epoch_weight_pde / n_steps_this_epoch

        # Let the alpha scheduler see epoch-level averages
        alpha_scheduler.on_epoch_end(avg_bc_loss, avg_pde_loss)

        # Print scheduler messages (e.g. alpha plateau increase)
        msg = getattr(alpha_scheduler, 'message', None)
        if msg:
            print(f"\n>>> {msg} <<<")
            best_val_loss = float('inf')
            epochs_without_improvement = 0

        # -- test / validation -----------------------------------------------
        with torch.no_grad():
            test_loss = model.loss_boundary(x_test, y_test)

        if use_validation:
            with torch.no_grad():
                val_bc_loss = model.loss_boundary(x_bc_val, y_bc_val)
                val_loss = val_bc_loss

            history['val_loss'].append(val_loss.item())
            history['val_bc_loss'].append(val_bc_loss.item())
            history['val_pde_loss'].append(None)

            if val_loss.item() < best_val_loss and alpha_scheduler.in_final_stage:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                if output_dirs:
                    ckpt = os.path.join(output_dirs['checkpoints'],
                                        'best_model.pt')
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_model_state,
                        'val_loss': best_val_loss,
                    }, ckpt)
            elif alpha_scheduler.in_final_stage:
                epochs_without_improvement += 1
        else:
            history['val_loss'].append(None)
            history['val_bc_loss'].append(None)
            history['val_pde_loss'].append(None)

        # -- record history --------------------------------------------------
        _, cur_alpha = alpha_scheduler.get_weights()

        history['total_loss'].append(avg_loss)
        history['bc_loss'].append(avg_bc_loss)
        history['pde_loss'].append(avg_pde_loss)
        history['interface_loss'].append(avg_intf_loss)
        history['test_loss'].append(test_loss.item())
        history['step'].append(global_step)
        history['weight_bc'].append(avg_weight_bc)
        history['weight_pde'].append(avg_weight_pde)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        history['alpha'].append(cur_alpha)

        # -- epoch log -------------------------------------------------------
        val_str = f"Val: {val_loss.item():.4e}" if use_validation else ""
        best_str = " *" if use_validation and epoch + 1 == best_epoch else ""
        lr_str = f"LR={current_lr:.1e}" if lr_scheduler is not None else ""
        intf_str = f"Intf: {avg_intf_loss:.4e} | " if avg_intf_loss > 0 else ""

        print(f"Epoch {epoch+1:3d}/{n_epochs} | Steps: {global_step:6d} | "
              f"{alpha_scheduler.log_str} | Loss: {avg_loss:.4e} | "
              f"BC: {avg_bc_loss:.4e} | PDE: {avg_pde_loss:.4e} | "
              f"{intf_str}{val_str} | Test: {test_loss.item():.4e} | "
              f"{lr_str}{best_str}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        # PDE-loss diagnostic plot
        if plot_pde_every > 0 and (epoch + 1) % plot_pde_every == 0:
            plots_dir = output_dirs['plots'] if output_dirs else '.'
            plot_pde_loss_distribution(model, x_interior_epoch, epoch + 1,
                                       example_name, output_dir=plots_dir,
                                       device=device)

        # Early stopping
        if patience and use_validation and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

        # Loss-threshold stopping (from experiment 6_2)
        if bc_loss_threshold is not None or pde_loss_threshold is not None:
            bc_met = True
            pde_met = True
            if bc_loss_threshold is not None:
                check_bc = val_bc_loss.item() if use_validation else avg_bc_loss
                bc_met = check_bc < bc_loss_threshold
            if pde_loss_threshold is not None:
                pde_met = avg_pde_loss < pde_loss_threshold

            if bc_loss_threshold is not None and pde_loss_threshold is not None:
                if bc_met and pde_met:
                    src = "val" if use_validation else "train"
                    print(f"\nBoth thresholds reached -- BC ({src}): {check_bc:.4e} < "
                          f"{bc_loss_threshold:.4e}, PDE: {avg_pde_loss:.4e} < "
                          f"{pde_loss_threshold:.4e}")
                    break
            elif bc_loss_threshold is not None and bc_met:
                src = "val" if use_validation else "train"
                print(f"\nBC loss threshold reached ({src}): {check_bc:.4e} < "
                      f"{bc_loss_threshold:.4e}")
                break
            elif pde_loss_threshold is not None and pde_met:
                print(f"\nPDE loss threshold reached: {avg_pde_loss:.4e} < "
                      f"{pde_loss_threshold:.4e}")
                break

    # -- post-training -------------------------------------------------------
    training_time = time.time() - start_time

    if use_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            best_test = torch.mean((model(x_test) - y_test) ** 2).item()
        model.train()
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(val_loss: {best_val_loss:.4e}, test_loss: {best_test:.4e})")
        if output_dirs:
            print(f"Best checkpoint saved to: "
                  f"{os.path.join(output_dirs['checkpoints'], 'best_model.pt')}")

    if output_dirs:
        final_path = os.path.join(output_dirs['checkpoints'],
                                  'final_model.pt')
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'history': history,
        }, final_path)
        print(f"Final checkpoint saved to: {final_path}")

    print(f"Training completed in {training_time:.2f}s ({global_step:,} total steps)")

    return history, training_time


# ============================================================================
# Iterative training (progressive p-ramping for p-Laplacian)
# ============================================================================

def train_pinn_iterative(
    model,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    x_interior: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    p_values: List[float],
    epochs_per_p: int = 50,
    batch_size_bc: int = 400,
    batch_size_pde: int = 1000,
    lr: float = 1e-3,
    alpha_scheduler=None,
    x_bc_val: Optional[torch.Tensor] = None,
    y_bc_val: Optional[torch.Tensor] = None,
    x_interior_val: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    interior_sampler=None,
    scheduler_type: Optional[str] = None,
    scheduler_end_factor: float = 0.01,
    plot_pde_every: int = 0,
    example_name: str = '',
    output_dirs: Optional[dict] = None,
    outlier_off_epoch: Optional[int] = None,
    original_outlier_pct: Optional[float] = None,
    bc_loss_threshold: Optional[float] = None,
    pde_loss_threshold: Optional[float] = None,
    resume_from_p: Optional[float] = None,
    seed: int = 1234,
    device: Optional[torch.device] = None,
):
    """
    Iterative training: call ``train_pinn`` for each p-value in sequence.

    The model must expose a ``p`` attribute (``model.p``) which is set
    before each stage.  Alpha schedule / outlier removal reset per stage.

    Returns:
        ``(combined_history, total_time)``
    """
    if device is None:
        device = get_device()

    if resume_from_p is not None:
        remaining = [pv for pv in p_values if pv >= resume_from_p]
        skipped = len(p_values) - len(remaining)
        print(f"\nResuming iterative training from p={resume_from_p}")
        print(f"  Skipping {skipped} completed stage(s), {len(remaining)} remaining")
        print(f"  Remaining p values: {remaining}")
    else:
        remaining = list(p_values)

    print(f"\nIterative Training Schedule:")
    print(f"  p values: {p_values}")
    print(f"  Epochs per p: {epochs_per_p}")
    print()

    if original_outlier_pct is None:
        original_outlier_pct = getattr(model, 'outlier_percentile', 0.0)

    combined_history = {
        'total_loss': [], 'bc_loss': [], 'pde_loss': [],
        'interface_loss': [],
        'val_loss': [], 'val_bc_loss': [], 'val_pde_loss': [],
        'test_loss': [], 'step': [],
        'weight_bc': [], 'weight_pde': [], 'lr': [], 'alpha': [],
        'p': [], 'p_schedule': list(p_values),
    }

    total_time = 0.0

    for p_idx, p in enumerate(p_values):
        if p not in remaining:
            continue
        overall_idx = p_values.index(p)
        print(f"\n{'=' * 60}")
        print(f"Stage {overall_idx + 1}/{len(p_values)}: Training with p={p}")
        print(f"{'=' * 60}")

        model.p = p
        model.outlier_percentile = original_outlier_pct
        if hasattr(alpha_scheduler, 'reset'):
            alpha_scheduler.reset()

        stage_name = f"p{p}"

        stage_history, stage_time = train_pinn(
            model, x_bc, y_bc, x_interior, x_test, y_test,
            n_epochs=epochs_per_p,
            batch_size_bc=batch_size_bc,
            batch_size_pde=batch_size_pde,
            lr=lr,
            alpha_scheduler=alpha_scheduler,
            x_bc_val=x_bc_val,
            y_bc_val=y_bc_val,
            x_interior_val=x_interior_val,
            patience=patience,
            interior_sampler=interior_sampler,
            scheduler_type=scheduler_type,
            scheduler_end_factor=scheduler_end_factor,
            plot_pde_every=plot_pde_every,
            example_name=stage_name,
            output_dirs=output_dirs,
            outlier_off_epoch=outlier_off_epoch,
            bc_loss_threshold=bc_loss_threshold,
            pde_loss_threshold=pde_loss_threshold,
            seed=seed,
            device=device,
        )

        total_time += stage_time

        if output_dirs:
            ckpt_dir = output_dirs['checkpoints']
            for base_name in ('best_model.pt', 'final_model.pt'):
                src = os.path.join(ckpt_dir, base_name)
                if os.path.exists(src):
                    dst = os.path.join(ckpt_dir,
                                       base_name.replace('.pt', f'_p{p}.pt'))
                    shutil.copy2(src, dst)

        # Per-stage PDE test evaluation
        model.eval()
        x_test_dev = x_test.to(device)
        with torch.no_grad() if not any(
            hasattr(model, m) for m in ['loss_pde_p_raw']
        ) else torch.enable_grad():
            pde_lines = [f"  Stage {overall_idx + 1} (p={p}) PDE diagnostics:"]
            if hasattr(model, 'loss_pde_p_raw'):
                raw = model.loss_pde_p_raw(x_test_dev, p).item()
                pde_lines.append(f"    p-Lap (p={p}) raw:        {raw:.4e}")
            if hasattr(model, 'loss_pde_p_normalized'):
                norm_p = model.loss_pde_p_normalized(x_test_dev, p).item()
                pde_lines.append(f"    p-Lap (p={p}) normalized: {norm_p:.4e}")
            if hasattr(model, 'loss_pde_infinity'):
                inf_raw = model.loss_pde_infinity(x_test_dev).item()
                pde_lines.append(f"    Inf-Lap raw:              {inf_raw:.4e}")
            if hasattr(model, 'loss_pde_infinity_normalized'):
                inf_norm = model.loss_pde_infinity_normalized(x_test_dev).item()
                pde_lines.append(f"    Inf-Lap normalized:       {inf_norm:.4e}")
            if len(pde_lines) > 1:
                print('\n'.join(pde_lines))

        # Append p value for each epoch in this stage
        n_stage_epochs = len(stage_history['total_loss'])
        stage_history.setdefault('p', [p] * n_stage_epochs)
        if len(stage_history['p']) < n_stage_epochs:
            stage_history['p'] = [p] * n_stage_epochs

        for key in combined_history:
            if key == 'p_schedule':
                continue
            if key in stage_history:
                combined_history[key].extend(stage_history[key])

    if output_dirs:
        torch.save({
            'epoch': len(combined_history['total_loss']),
            'model_state_dict': model.state_dict(),
            'history': combined_history,
        }, os.path.join(output_dirs['checkpoints'], 'final_model.pt'))

    print(f"\nIterative training completed in {total_time:.2f}s")
    print(f"Total epochs: {len(combined_history['total_loss'])}")

    return combined_history, total_time
